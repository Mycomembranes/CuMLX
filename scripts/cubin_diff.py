#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: structural validator + equality GATE for cubins
# produced by ptxas-darwin-arm64 (solution 105, Route B) against Linux ptxas
# references. Pure stdlib; no CUDA needed.
#
#   cubin_diff.py check A.cubin [--expect-sm 120] [--ptx A.ptx]   # structural validation
#   cubin_diff.py diff  NATIVE.cubin LINUX.cubin [--json] [--strict-options]
#
# `check` validates: ET_EXEC, EM_CUDA, the expected SM, at least one entry, a non-empty
# .text per entry, strict .nv.info record parsing (truncated/overrunning records and
# unknown formats are errors), well-formed KPARAM_INFO (12-byte payload, ordinals
# contiguous from 0, non-zero sizes, increasing non-overlapping offsets inside
# CBANK_PARAM_SIZE) and, with --ptx, the same entry set and per-parameter sizes as the
# PTX `.entry` declarations.
#
# `diff` is a gate: it exits 1 on ANY difference except the documented
# `.note.nv.tkinfo` tolerances, and 0 only when the cubins are
# "section-identical except .note.nv.tkinfo". Compared exactly:
#   * the ELF header (all 64 bytes; e_phoff/e_shoff via the layout check below)
#   * the section list, in order, and every section header field
#     (sh_size of .note.nv.tkinfo and sh_offset via the layout check)
#   * every section's bytes except .note.nv.tkinfo
#   * every program header field (p_offset via the layout check)
#   * LAYOUT: file offsets of every section, the program and section header tables and
#     every p_offset must equal the reference's after re-laying the file out with the
#     native .note.nv.tkinfo size (same order, same alignment, same extra gaps): the
#     only shift allowed is the one that note's length change induces. Bytes outside
#     the ELF header, both header tables and all section data must be zero.
#   * .note.nv.tkinfo, parsed: note count, namesz, type, name, desc format version,
#     object-file name, tool name ("ptxas"), the string-table layout and the zero padding,
#     and descsz: the used-length difference must equal exactly the length difference
#     of the tolerated strings, with < 4 bytes of zero padding.
#     Tolerated (and only these): the tool-version string (must still read
#     "Cuda compilation tools, release X.Y, V..." with the SAME release X.Y), the
#     build string (Linux "Build cuda_13.2.r13.2/compiler.<n>" vs the static library's
#     "Build system must define TOOLS_VERSION_EXTENDED"), and the echoed option
#     string. --strict-options also requires the option strings to be equal
#     (milestone.sh uses it: it compiles with the reference's own options).
# Exit codes: 0 gate PASS, 1 gate FAIL (or check problems), 2 usage / unreadable input.
import argparse
import hashlib
import json
import re
import struct
import sys

EM_CUDA = 190
TKINFO = ".note.nv.tkinfo"
TOOL_VERSION_RE = re.compile(r"^Cuda compilation tools, release (\d+\.\d+), V\d+\.\d+\.\d+$")
EIATTR = {0x17: "KPARAM_INFO", 0x0a: "PARAM_CBANK", 0x19: "CBANK_PARAM_SIZE", 0x1b: "MAXREG_COUNT",
          0x1c: "EXIT_INSTR_OFFSETS", 0x2f: "REGCOUNT", 0x50: "CUDA_API_VERSION", 0x37: "SW_WAR"}


class CubinError(Exception):
    pass


def parse(path):
    try:
        with open(path, "rb") as f:
            b = f.read()
    except OSError as e:
        raise CubinError(f"{path}: {e}") from e
    if len(b) < 64 or b[:4] != b"\x7fELF" or b[4] != 2 or b[5] != 1:
        raise CubinError(f"{path}: not a little-endian ELF64")
    e_type, machine = struct.unpack_from("<HH", b, 16)
    e_flags, = struct.unpack_from("<I", b, 48)
    phoff, shoff = struct.unpack_from("<QQ", b, 32)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", b, 54)
    if shoff + shnum * shentsize > len(b) or phoff + phnum * phentsize > len(b) or shstrndx >= max(shnum, 1):
        raise CubinError(f"{path}: truncated section/program header table")
    if shentsize != 64 or (phnum and phentsize != 56):
        raise CubinError(f"{path}: unexpected header entry sizes sh={shentsize} ph={phentsize}")
    secs = []
    for i in range(shnum):
        nm, ty, fl, addr, off, size, link, info, al, es = struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)
        secs.append(dict(nm=nm, type=ty, flags=fl, addr=addr, off=off, size=size, link=link, info=info,
                         align=al, entsize=es))
    stro, strsz = secs[shstrndx]["off"], secs[shstrndx]["size"]
    if stro + strsz > len(b):
        raise CubinError(f"{path}: .shstrtab out of bounds")
    for s in secs:
        if s["nm"] >= strsz:
            raise CubinError(f"{path}: section name offset {s['nm']} outside .shstrtab ({strsz} B)")
        e = b.find(b"\0", stro + s["nm"], stro + strsz)
        if e < 0:
            raise CubinError(f"{path}: unterminated section name at {s['nm']}")
        s["name"] = b[stro + s["nm"]:e].decode("latin-1")
        if s["type"] != 8 and s["off"] + s["size"] > len(b):
            raise CubinError(f"{path}: section {s['name']} out of bounds")
        s["data"] = b"" if s["type"] == 8 else b[s["off"]:s["off"] + s["size"]]
    phdrs = []
    for i in range(phnum):
        ty, fl, off, va, pa, fs, ms, al = struct.unpack_from("<IIQQQQQQ", b, phoff + i * phentsize)
        phdrs.append(dict(type=ty, flags=fl, offset=off, vaddr=va, paddr=pa, filesz=fs, memsz=ms, align=al))
    abiv = b[8]
    sm = (e_flags >> 8) & 0xff if abiv >= 8 else e_flags & 0xff
    return dict(path=path, bytes=b, e_type=e_type, machine=machine, e_flags=e_flags, abiv=abiv, sm=sm, secs=secs,
                phdrs=phdrs, phoff=phoff, shoff=shoff, phnum=phnum, shnum=shnum)


EIFMT_NVAL, EIFMT_BVAL, EIFMT_HVAL, EIFMT_SVAL = 0x01, 0x02, 0x03, 0x04
EIATTR_KPARAM_INFO, EIATTR_CBANK_PARAM_SIZE = 0x17, 0x19


def nv_info(data):
    """Strict EIATTR walk -> [(fmt, attr, value_or_payload)]. Every record is 4 bytes
    (fmt, attr, u16) except EIFMT_SVAL, whose u16 is the payload length. A record that
    overruns the section, trailing bytes, or an unknown format raise CubinError."""
    out, i = [], 0
    while i < len(data):
        if i + 4 > len(data):
            raise CubinError(f".nv.info: truncated record header at {i}")
        fmt, attr = data[i], data[i + 1]
        v, = struct.unpack_from("<H", data, i + 2)
        if fmt == EIFMT_SVAL:
            if i + 4 + v > len(data):
                raise CubinError(f".nv.info: attr 0x{attr:02x} payload ({v} B) overruns the section at {i}")
            out.append((fmt, attr, data[i + 4:i + 4 + v]))
            i += 4 + v
        elif fmt in (EIFMT_NVAL, EIFMT_BVAL, EIFMT_HVAL):
            out.append((fmt, attr, v))
            i += 4
        else:
            raise CubinError(f".nv.info: unknown record format 0x{fmt:02x} at {i}")
    return out


PTX_TYPE_BYTES = {"8": 1, "16": 2, "32": 4, "64": 8, "128": 16}


def ptx_entries(text):
    """{entry name: [param byte sizes]} from PTX `.entry name( .param ... )` declarations."""
    import re
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    out = {}
    for m in re.finditer(r"\.entry\s+([A-Za-z_$%][\w$%]*)\s*\(", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        if depth:
            raise CubinError(f"PTX: unterminated parameter list of {m.group(1)}")
        params = [p.strip() for p in text[m.end():j - 1].split(",") if p.strip()]
        sizes = []
        for p in params:
            t = re.search(r"\.[bsuf](8|16|32|64|128)\b", p)
            if not p.startswith(".param") or not t:
                raise CubinError(f"PTX: cannot size parameter {p!r} of {m.group(1)}")
            n = re.search(r"\[(\d+)\]\s*$", p)
            sizes.append(PTX_TYPE_BYTES[t.group(1)] * (int(n.group(1)) if n else 1))
        if m.group(1) in out:
            raise CubinError(f"PTX: duplicate entry {m.group(1)}")
        out[m.group(1)] = sizes
    return out


def kparams(info_data, kname):
    """Validate one .nv.info.<k> and return its KPARAM list [(ordinal, offset, size)] by ordinal."""
    recs = nv_info(info_data)
    params, cbank_param_size = [], None
    for fmt, attr, v in recs:
        if attr == EIATTR_KPARAM_INFO:
            if fmt != EIFMT_SVAL or len(v) != 12:
                raise CubinError(f"{kname}: KPARAM_INFO record malformed (fmt {fmt}, {len(v) if fmt == EIFMT_SVAL else '-'} B)")
            _idx, ordinal, offset, flags = struct.unpack("<IHHI", v)
            params.append((ordinal, offset, (flags >> 18) & 0x3fff))
        elif attr == EIATTR_CBANK_PARAM_SIZE and fmt == EIFMT_HVAL:
            cbank_param_size = v
    params.sort()
    if [o for o, _, _ in params] != list(range(len(params))):
        raise CubinError(f"{kname}: KPARAM ordinals {[o for o, _, _ in params]} not contiguous from 0")
    end = 0
    for o, off, size in params:
        if size == 0:
            raise CubinError(f"{kname}: KPARAM {o} has size 0")
        if off < end:
            raise CubinError(f"{kname}: KPARAM {o} at offset {off} overlaps the previous parameter (ends {end})")
        end = off + size
    if params and cbank_param_size is not None and end > cbank_param_size:
        raise CubinError(f"{kname}: parameters end at {end} > CBANK_PARAM_SIZE {cbank_param_size}")
    if params and cbank_param_size is None:
        raise CubinError(f"{kname}: KPARAM_INFO without CBANK_PARAM_SIZE")
    return params


def parse_tkinfo(data):
    """Parse .note.nv.tkinfo: ELF notes whose desc is
    {u32 format, u32 off_objname, off_toolname, off_toolversion, off_toolbranch, off_tooloptions}
    followed by a NUL-separated string table. Raises CubinError on malformed input."""
    notes, i = [], 0
    while i + 12 <= len(data):
        namesz, descsz, ntype = struct.unpack_from("<III", data, i)
        npad = (namesz + 3) & ~3
        dpad = (descsz + 3) & ~3
        if i + 12 + npad + dpad > len(data):
            raise CubinError("tkinfo: note overruns section")
        name = data[i + 12:i + 12 + namesz]
        desc = data[i + 12 + npad:i + 12 + npad + descsz]
        pad = data[i + 12 + namesz:i + 12 + npad] + data[i + 12 + npad + descsz:i + 12 + npad + dpad]
        if len(desc) < 24:
            raise CubinError("tkinfo: desc shorter than its 24-byte header")
        fmt, o_obj, o_tool, o_ver, o_branch, o_opts = struct.unpack_from("<6I", desc, 0)
        strtab = desc[24:]

        def s_at(o, what):
            if o >= len(strtab):
                raise CubinError(f"tkinfo: {what} offset {o} outside string table ({len(strtab)} B)")
            if o and strtab[o - 1] != 0:
                raise CubinError(f"tkinfo: {what} offset {o} is not at a string boundary")
            e = strtab.index(b"\0", o)
            return strtab[o:e].decode("latin-1")

        fields = dict(objname=s_at(o_obj, "objname"), tool=s_at(o_tool, "tool"), version=s_at(o_ver, "version"),
                      build=s_at(o_branch, "build"), options=s_at(o_opts, "options"))
        # String-table layout: the referenced strings, in offset order, must tile the table
        # exactly (no hidden bytes); anything after the last string must be NUL padding.
        offs = sorted({o_obj, o_tool, o_ver, o_branch, o_opts})
        end = strtab.index(b"\0", offs[-1]) + 1
        rest = strtab[end:]
        layout_ok = all(strtab.index(b"\0", a) + 1 == b for a, b in zip(offs, offs[1:])) and offs[0] in (0, 1) \
            and (offs[0] == 0 or strtab[0] == 0)
        used = 24 + end
        if descsz - used > 3:
            layout_ok = False
        notes.append(dict(namesz=namesz, type=ntype, name=name.hex(), format=fmt, fields=fields,
                          descsz=descsz, used=used,
                          order=[k for _, k in sorted([(o_obj, "objname"), (o_tool, "tool"), (o_ver, "version"),
                                                       (o_branch, "build"), (o_opts, "options")])],
                          layout_ok=layout_ok, padding_zero=not any(pad) and not any(rest)))
        i += 12 + npad + dpad
    if any(data[i:]):
        raise CubinError("tkinfo: trailing non-zero bytes after the last note")
    if not notes:
        raise CubinError("tkinfo: no notes")
    return notes


def compare_tkinfo(da, db, strict_options=False):
    """Return (violations, tolerated) for two tkinfo payloads."""
    v, tol = [], []
    try:
        na, nb = parse_tkinfo(da), parse_tkinfo(db)
    except CubinError as e:
        return [f"{TKINFO}: {e}"], tol
    if len(na) != len(nb):
        return [f"{TKINFO}: note count {len(na)} != {len(nb)}"], tol
    for k, (a, b) in enumerate(zip(na, nb)):
        p = f"{TKINFO}[{k}]"
        for f in ("namesz", "type", "name", "format", "order"):
            if a[f] != b[f]:
                v.append(f"{p}: {f} {a[f]!r} != {b[f]!r}")
        for f in ("objname", "tool"):
            if a["fields"][f] != b["fields"][f]:
                v.append(f"{p}: {f} {a['fields'][f]!r} != {b['fields'][f]!r}")
        for side, n in (("native", a), ("linux", b)):
            if not n["layout_ok"]:
                v.append(f"{p}: {side} string table has unreferenced bytes")
            if not n["padding_zero"]:
                v.append(f"{p}: {side} padding is not zero")
        tol_delta = sum(len(a["fields"][f]) - len(b["fields"][f]) for f in ("version", "build", "options"))
        if a["used"] - b["used"] != tol_delta:
            v.append(f"{p}: descsz {a['descsz']} vs {b['descsz']}: used length differs by "
                     f"{a['used'] - b['used']}, tolerated strings explain {tol_delta}")
        va, vb = a["fields"]["version"], b["fields"]["version"]
        ma, mb = TOOL_VERSION_RE.match(va), TOOL_VERSION_RE.match(vb)
        if not ma or not mb:
            v.append(f"{p}: tool version not in 'Cuda compilation tools, release X.Y, VX.Y.Z' form: {va!r} / {vb!r}")
        elif ma.group(1) != mb.group(1):
            v.append(f"{p}: CUDA release {ma.group(1)} != {mb.group(1)}")
        elif va != vb:
            tol.append(f"{p}: version {va!r} vs {vb!r}")
        if a["fields"]["build"] != b["fields"]["build"]:
            tol.append(f"{p}: build {a['fields']['build']!r} vs {b['fields']['build']!r}")
        oa, ob = a["fields"]["options"], b["fields"]["options"]
        if oa != ob:
            if strict_options:
                v.append(f"{p}: options {oa!r} != {ob!r} (--strict-options)")
            else:
                tol.append(f"{p}: options {oa!r} vs {ob!r}")
    return v, tol


def kernels(c):
    return [s["name"][len(".text."):] for s in c["secs"] if s["name"].startswith(".text.")]


def check(path, expect_sm, ptx=None):
    c = parse(path)
    problems = []
    if c["e_type"] != 2:
        problems.append(f"e_type {c['e_type']} != ET_EXEC(2)")
    if c["machine"] != EM_CUDA:
        problems.append(f"e_machine {c['machine']} != EM_CUDA(190)")
    if expect_sm and c["sm"] != expect_sm:
        problems.append(f"sm {c['sm']} != {expect_sm} (e_flags=0x{c['e_flags']:x})")
    ks = kernels(c)
    if not ks:
        problems.append("no .text.<kernel> sections")
    by = {s["name"]: s for s in c["secs"]}
    per = {}
    for k in ks:
        text = by[f".text.{k}"]
        if not text["data"]:
            problems.append(f"{k}: empty .text")
        info = by.get(f".nv.info.{k}")
        kp = []
        if info is None:
            problems.append(f"{k}: missing .nv.info.{k}")
        else:
            try:
                kp = kparams(info["data"], k)
            except CubinError as e:
                problems.append(str(e))
        if f".nv.constant0.{k}" not in by:
            problems.append(f"{k}: missing .nv.constant0.{k}")
        per[k] = dict(kparams=len(kp), param_sizes=[sz for _, _, sz in kp], text_bytes=len(text["data"]))
    for s in c["secs"]:
        if s["name"] == ".nv.info":
            try:
                nv_info(s["data"])
            except CubinError as e:
                problems.append(f".nv.info: {e}")
    if ptx is not None:
        try:
            with open(ptx, "r", encoding="latin-1") as f:
                ents = ptx_entries(f.read())
        except (OSError, CubinError) as e:
            problems.append(f"PTX {ptx}: {e}")
            ents = None
        if ents is not None:
            if set(ents) != set(per):
                problems.append(f"entries differ from PTX: cubin-only={sorted(set(per) - set(ents))} "
                                f"ptx-only={sorted(set(ents) - set(per))}")
            for k in sorted(set(ents) & set(per)):
                if per[k]["param_sizes"] != ents[k]:
                    problems.append(f"{k}: KPARAM sizes {per[k]['param_sizes']} != PTX params {ents[k]}")
    return dict(path=path, machine=c["machine"], sm=c["sm"], e_flags=hex(c["e_flags"]), abi_version=c["abiv"],
                kernels=per, sections=len(c["secs"]), ok=not problems, problems=problems)


SHDR_FIELDS = ("type", "flags", "addr", "link", "info", "align", "entsize")
PHDR_FIELDS = ("type", "flags", "vaddr", "paddr", "filesz", "memsz", "align")


def _align_up(x, a):
    return x if a <= 1 else (x + a - 1) // a * a


def layout_entries(c):
    """File-layout entries in file order: (key, offset, file size, alignment)."""
    ents = [(("sec", i, s["name"]), s["off"], 0 if s["type"] == 8 else s["size"], max(s["align"], 1))
            for i, s in enumerate(c["secs"]) if s["type"] != 0]
    ents.append((("shdrs",), c["shoff"], c["shnum"] * 64, 8))
    if c["phnum"]:
        ents.append((("phdrs",), c["phoff"], c["phnum"] * 56, 8))
    return sorted(ents, key=lambda e: (e[1], e[0][1] if e[0][0] == "sec" else 1 << 30))


def check_layout(A, B):
    """Return (violations, shift_of_key) -- re-lay A out in B's order with A's sizes."""
    v = []
    ea = {e[0]: e for e in layout_entries(A)}
    eb = layout_entries(B)
    if set(ea) != {e[0] for e in eb}:
        return ["layout: different set of sections/header tables"], {}
    shift = {}
    end_a = end_b = 64
    for key, off_b, size_b, al in eb:
        _, off_a, size_a, _ = ea[key]
        extra = off_b - _align_up(end_b, al)
        expect = _align_up(end_a, al) + extra if extra >= 0 else end_a + (off_b - end_b)
        if off_a != expect:
            v.append(f"layout: {key[-1] if key[0] == 'sec' else key[0]} at {off_a:#x}, expected {expect:#x} "
                     f"(reference {off_b:#x}; only the .note.nv.tkinfo length change may shift offsets)")
        shift[key] = off_a - off_b
        end_a = max(end_a, off_a + size_a)
        end_b = max(end_b, off_b + size_b)
    if len(A["bytes"]) - end_a != len(B["bytes"]) - end_b:
        v.append(f"layout: trailing bytes {len(A['bytes']) - end_a} != {len(B['bytes']) - end_b}")
    return v, shift


def uncovered_nonzero(c):
    """Offsets of non-zero bytes not covered by the ELF header, header tables or section data."""
    cov = bytearray(len(c["bytes"]))
    spans = [(0, 64)] + [(e[1], e[1] + e[2]) for e in layout_entries(c)]
    for a, b in spans:
        cov[a:b] = b"\1" * (min(b, len(cov)) - a) if a < len(cov) else b""
    return [i for i, (x, m) in enumerate(zip(c["bytes"], cov)) if x and not m]


def phdr_expected_offset(p_off, B, shift):
    for key, off, size, _ in layout_entries(B):
        if off == p_off and size:
            return p_off + shift[key]
    for key, off, size, _ in layout_entries(B):
        if off <= p_off < off + max(size, 1):
            return p_off + shift[key]
    return p_off


def diff(a, b, strict_options=False):
    A, B = parse(a), parse(b)
    violations, tolerated, rows = [], [], []
    # ELF header: everything but e_phoff (32..40) and e_shoff (40..48), which the
    # layout check below validates.
    ha, hb = A["bytes"][:64], B["bytes"][:64]
    if ha[:32] != hb[:32] or ha[48:64] != hb[48:64]:
        violations.append("ELF header differs (outside e_phoff/e_shoff)")
    lv, shift = check_layout(A, B)
    violations += lv
    for side, c in (("native", A), ("linux", B)):
        nz = uncovered_nonzero(c)
        if nz:
            violations.append(f"{side}: {len(nz)} non-zero byte(s) outside all sections/headers (first at {nz[0]:#x})")
    names_a = [s["name"] for s in A["secs"]]
    names_b = [s["name"] for s in B["secs"]]
    if names_a != names_b:
        violations.append(f"section list/order differs: only-native={sorted(set(names_a) - set(names_b))} "
                          f"only-linux={sorted(set(names_b) - set(names_a))}")
    an = {s["name"]: s for s in A["secs"]}
    bn = {s["name"]: s for s in B["secs"]}
    for name in list(dict.fromkeys(names_a + names_b)):
        sa, sb = an.get(name), bn.get(name)
        if sa is None or sb is None:
            rows.append(dict(section=name, status="only-in-" + ("native" if sb is None else "linux")))
            continue
        for f in SHDR_FIELDS:
            if sa[f] != sb[f]:
                violations.append(f"{name or '<null>'}: sh_{f} {sa[f]:#x} != {sb[f]:#x}")
        da, db = sa["data"], sb["data"]
        nd = sum(1 for x, y in zip(da, db) if x != y) + abs(len(da) - len(db))
        status = "identical" if da == db else f"differs ({nd} bytes)"
        if name == TKINFO:
            v, t = compare_tkinfo(da, db, strict_options)
            violations += v
            tolerated += t
            if da != db:
                status += " [tolerated fields only]" if not v else " [VIOLATION]"
        else:
            if sa["size"] != sb["size"]:
                violations.append(f"{name}: sh_size {sa['size']} != {sb['size']}")
            if da != db:
                violations.append(f"{name}: {status}")
        rows.append(dict(section=name, size_native=len(da), size_linux=len(db), status=status,
                         sha_native=hashlib.sha256(da).hexdigest()[:16], sha_linux=hashlib.sha256(db).hexdigest()[:16]))
    if len(A["phdrs"]) != len(B["phdrs"]):
        violations.append(f"program header count {len(A['phdrs'])} != {len(B['phdrs'])}")
    for k, (pa, pb) in enumerate(zip(A["phdrs"], B["phdrs"])):
        for f in PHDR_FIELDS:
            if pa[f] != pb[f]:
                violations.append(f"phdr[{k}]: p_{f} {pa[f]:#x} != {pb[f]:#x}")
        if shift:
            exp = phdr_expected_offset(pb["offset"], B, shift)
            if pa["offset"] != exp:
                violations.append(f"phdr[{k}]: p_offset {pa['offset']:#x}, expected {exp:#x} after layout shift")
    texts = [r for r in rows if r["section"].startswith(".text.")]
    return dict(native=a, linux=b, elf_header_identical=ha == hb, file_identical=A["bytes"] == B["bytes"],
                all_text_identical=bool(texts) and all(r["status"] == "identical" for r in texts), sections=rows,
                gate="PASS" if not violations else "FAIL", violations=violations, tolerated=tolerated,
                strict_options=strict_options)


def main():
    ap = argparse.ArgumentParser(description="cubin structural check + section-equality gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("cubin", nargs="+")
    c.add_argument("--expect-sm", type=int, default=120)
    c.add_argument("--ptx", help="validate entries + KPARAM sizes against this PTX (single cubin)")
    d = sub.add_parser("diff")
    d.add_argument("native")
    d.add_argument("linux")
    d.add_argument("--json", action="store_true")
    d.add_argument("--strict-options", action="store_true",
                   help="also require equal tkinfo option strings")
    a = ap.parse_args()
    try:
        if a.cmd == "check":
            if a.ptx and len(a.cubin) != 1:
                print("cubin_diff: --ptx takes exactly one cubin", file=sys.stderr)
                return 2
            res = [check(p, a.expect_sm, a.ptx) for p in a.cubin]
            print(json.dumps(res, indent=1))
            return 0 if all(r["ok"] for r in res) else 1
        r = diff(a.native, a.linux, a.strict_options)
    except CubinError as e:
        print(f"cubin_diff: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"native {r['native']}\nlinux  {r['linux']}\nfile identical: {r['file_identical']}  "
              f"ELF header identical: {r['elf_header_identical']}  all .text identical: {r['all_text_identical']}")
        for s in r["sections"]:
            if "size_native" in s:
                print(f"  {s['section']:<40} {s['size_native']:>6} {s['size_linux']:>6}  {s['status']}")
            else:
                print(f"  {s['section']:<40} {s['status']}")
        for t in r["tolerated"]:
            print(f"  tolerated: {t}")
        for v in r["violations"]:
            print(f"  VIOLATION: {v}")
        print(f"gate: {r['gate']} ({len(r['violations'])} violations; "
              f"section-identical except {TKINFO} version/build{'' if r['strict_options'] else '/options'})")
    return 0 if r["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
