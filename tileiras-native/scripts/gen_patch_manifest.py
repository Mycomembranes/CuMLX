#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: offline patch manifest for NVIDIA's Linux-aarch64
# `tileiras` ET_EXEC (adapted from portkit/ptxas-native/scripts/gen_x18_patches.py).
#
# The binary cannot run unmodified on Darwin arm64 for four independent reasons,
# each of which becomes one row class of the manifest:
#
#   x18    Linux AAPCS64 uses x18 as a temporary; Darwin zeroes user x18
#          asynchronously. Every instruction naming x18/w18 is listed with a
#          replacement word (x18 -> R) plus a second scratch register R2. The
#          loader moves the instruction into an island that keeps "virtual x18"
#          in a per-thread slot, so no liveness analysis is needed and no
#          function boundary has to be trusted.
#   tls    `mrs xN, tpidr_el0`: Darwin does not preserve a user TPIDR_EL0. The
#          loader replaces the read by an island that loads a Linux-style thread
#          pointer from a pthread TSD slot (TPIDRRO_EL0-relative).
#   svc    direct system calls (none are tolerated: any row is a hard error for
#          the loader, the count is recorded so the absence is checked).
#   abs64  8-byte absolute image addresses inside executable sections (literal
#          pools). Darwin arm64 forces a 4 GiB __PAGEZERO, so the fixed base
#          0x400000 is unmappable and the image is loaded at base + slide.
#
# Absolute pointers in data sections are covered by `# rebase` header lines
# (section range, word count, SHA-256 of the little-endian u64 link-time
# addresses of the rebased words); the loader recomputes and compares them.
#
#   norebase  an aligned in-range word of .data.rel.ro / .data that the rebase
#          rule must NOT slide because its value is contradicted as an address:
#          it lies in no section, inside unwind metadata, or inside an
#          executable section without being a function start (an FDE initial
#          location, a PLT entry, .init or .fini). Measured on the pinned
#          binary these are integers (bit masks 0x400000..0x4000000, the two
#          Itanium-demangler operator tables). This is evidence from the shape
#          of the value, not proof of its use. Left unslid, a wrong verdict
#          faults in the unmapped low 4 GiB when dereferenced or called and is
#          reported (not when only compared); slid, an integer would be
#          silently changed. `tileiras-native rebase-audit` derives the
#          same set independently (tests/helper_qc.rs compares them).
#
# The generator never executes the binary. Output is deterministic: the same
# ELF and the same llvm tools give a byte-identical manifest.
#
# Exit status: 0 = every site classified; 2 = refused / unsupported shape.
import argparse
import hashlib
import itertools
import os
import re
import struct
import subprocess
import sys

SCHEMA = "cuda-oxide.tileiras-native-patches.v1"
# Admitted inputs. ET_EXEC needs the rebase rule and absolute-literal rows;
# ET_DYN carries real R_AARCH64_RELATIVE relocations, so it gets neither.
PINS = {
    "297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a": "tileiras 13.2.51 linux-sbsa (ET_EXEC)",
    "f3cb2690f62c668924a373482a9bc748ace90b94bf13adce8e76a1f077682077": "libnvvm.so.4.0.0 13.2.51 linux-sbsa (ET_DYN)",
}
FIELDS = (0, 5, 10, 16)
CANDIDATES = (9, 10, 11, 12, 13, 14, 15)
GPR_RE = re.compile(r"\b[xw](\d{1,2})\b")
X18_RE = re.compile(r"\b([xw])18\b")
INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)\s*(.*)$")
# Sections whose aligned 8-byte words are link-time pointers when their value
# lies inside the image. .rodata is deliberately absent: the objects are PIC, so
# relocated pointers live in .data.rel.ro/.got; in-range .rodata words are
# integers (measured: 79 of 13,777 text-range hits are function starts = chance).
REBASE_SECTIONS = (".init_array", ".fini_array", ".data.rel.ro", ".got", ".got.plt", ".data")
# Sections whose every entry is an address by ELF typing: never second-guessed.
POINTER_ONLY_SECTIONS = (".init_array", ".fini_array", ".got", ".got.plt")
UNWIND_SECTIONS = (".eh_frame_hdr", ".eh_frame", ".gcc_except_table")


class Refuse(Exception):
    pass


def parse_elf(data):
    if data[:7] != b"\x7fELF\x02\x01\x01":
        raise Refuse("not ELF64 little-endian")
    kind, machine = struct.unpack_from("<HH", data, 16)
    if kind not in (2, 3) or machine != 183:
        raise Refuse("only AArch64 ET_EXEC / ET_DYN")
    phoff, shoff = struct.unpack_from("<QQ", data, 32)
    phnum = struct.unpack_from("<H", data, 56)[0]
    shnum, shstrndx = struct.unpack_from("<HH", data, 60)
    loads = []
    for i in range(phnum):
        t, f, off, va, _pa, fs, ms, _al = struct.unpack_from("<IIQQQQQQ", data, phoff + 56 * i)
        if t == 1:
            loads.append((va, off, fs, ms, f))
    raw = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + 64 * i) for i in range(shnum)]
    stroff = raw[shstrndx][4]
    secs = {}
    for s in raw:
        name = data[stroff + s[0]:data.index(b"\0", stroff + s[0])].decode("ascii")
        secs[name] = dict(type=s[1], flags=s[2], va=s[3], off=s[4], size=s[5])
    lo = min(va for va, *_ in loads)
    hi = max(va + ms for va, _o, _f, ms, _fl in loads)
    return loads, secs, lo, hi, kind == 3


def fde_ranges(data, secs):
    """Sorted [(start, end)] of every FDE in .eh_frame (pcrel|sdata4 only)."""
    sec = secs[".eh_frame"]
    base, off, size = sec["va"], sec["off"], sec["size"]
    pos, out, cies = 0, [], {}
    while pos + 4 <= size:
        length = struct.unpack_from("<I", data, off + pos)[0]
        if length == 0:
            break
        if length == 0xFFFFFFFF:
            raise Refuse("64-bit DWARF CFI is not supported")
        body = pos + 4
        cid = struct.unpack_from("<I", data, off + body)[0]
        if cid == 0:
            p = off + body + 5  # id + version
            aug_end = data.index(b"\0", p)
            aug = data[p:aug_end]
            p = aug_end + 1
            for _ in range(2):  # code/data alignment (uleb/sleb)
                while data[p] & 0x80:
                    p += 1
                p += 1
            while data[p] & 0x80:  # return address register
                p += 1
            p += 1
            enc = 0
            if aug.startswith(b"z"):
                while data[p] & 0x80:
                    p += 1
                p += 1
                for ch in aug[1:]:
                    if ch == ord("R"):
                        enc = data[p]
                        p += 1
                    elif ch == ord("L"):
                        p += 1
                    elif ch == ord("P"):
                        penc = data[p]
                        p += 1 + {0: 8, 2: 2, 3: 4, 4: 8, 0xB: 4, 0xC: 8}[penc & 0xF]
                    elif ch in (ord("S"), ord("B")):
                        pass
                    else:
                        raise Refuse(f"unknown CIE augmentation {aug!r}")
            cies[pos] = enc
        else:
            cie = body - cid
            enc = cies.get(cie)
            if enc != 0x1B:
                raise Refuse(f"FDE pointer encoding {enc!r} is not pcrel|sdata4")
            field = base + body + 4
            start = field + struct.unpack_from("<i", data, off + body + 4)[0]
            span = struct.unpack_from("<I", data, off + body + 8)[0]
            if span:
                out.append((start, start + span))
        pos = body + length
    out.sort()
    return out


class Coverage:
    def __init__(self, ranges):
        import bisect
        self._b = bisect
        merged = []
        for s, e in ranges:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.starts = [m[0] for m in merged]
        self.ends = [m[1] for m in merged]

    def __contains__(self, addr):
        i = self._b.bisect_right(self.starts, addr) - 1
        return i >= 0 and addr < self.ends[i]


def mc_disasm(mc, words):
    if not words:
        return []
    text = "\n".join(" ".join(f"0x{(w >> (8 * i)) & 0xff:02x}" for i in range(4)) for w in words)
    p = subprocess.run([mc, "--disassemble", "-triple=aarch64-linux-gnu", "-mattr=+all"],
                       input=text, capture_output=True, text=True)
    bad = {int(m.group(1)) - 1 for m in re.finditer(r"<stdin>:(\d+):\d+: warning: invalid instruction encoding", p.stderr)}
    lines = []
    for line in p.stdout.splitlines():
        t = line.strip()
        if not t or t.startswith(".text") or t.startswith("#"):
            continue
        lines.append(re.sub(r"\s+", " ", t.split("//")[0].strip()))
    if len(lines) + len(bad) != len(words):
        raise Refuse(f"llvm-mc returned {len(lines)} lines + {len(bad)} invalid for {len(words)} words")
    it = iter(lines)
    return [None if k in bad else next(it) for k in range(len(words))]


def x18_kind(mn, ops):
    if mn == "adrp":
        return "adrp"
    if mn in ("cbz", "cbnz"):
        return "cb"
    if mn in ("tbz", "tbnz"):
        return "tb"
    if mn == "blr":
        return "blr"
    if mn in ("br", "ret", "adr", "bl", "b") or mn.startswith("b."):
        return None  # no reviewed island shape
    if re.search(r"\b0x[0-9a-f]+ <", ops) or re.fullmatch(r".*,\s*0x[0-9a-f]+", ops) and mn.startswith("ldr") and "[" not in ops:
        return None  # pc-relative literal
    first = ops.split(",")[0].strip()
    if first in ("sp", "wsp") or re.search(r"\[sp[^\]]*\]!", ops) or re.search(r"\[sp\],", ops):
        return None  # writes sp: the island's red-zone slots would move
    return "plain"


def build(elf_path, llvm_bin):
    with open(elf_path, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    if sha not in PINS:
        raise Refuse(f"unadmitted input SHA256 {sha}")
    loads, secs, lo, hi, shared = parse_elf(data)
    exec_secs = [(n, s) for n, s in secs.items() if s["flags"] & 4 and s["size"]]
    exec_secs.sort(key=lambda t: t[1]["va"])
    fdes = fde_ranges(data, secs)
    cover = Coverage(fdes)
    plt = secs[".plt"]

    def in_code(addr):
        # .plt has no FDE but is linker-generated code; everything else must be
        # inside a function described by .eh_frame.
        return addr in cover or plt["va"] <= addr < plt["va"] + plt["size"]

    rows = []
    stats = dict(x18=0, tls=0, svc=0, abs64=0, norebase=0, x18_outside_fde=0, other_mrs=0)
    pending = []
    args = [os.path.join(llvm_bin, "llvm-objdump"), "-d", "--no-leading-headers"]
    for name, _ in exec_secs:
        args += ["-j", name]
    proc = subprocess.Popen(args + [elf_path], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if "18" not in line:
            continue
        m = INSN_RE.match(line)
        if not m:
            continue
        ops = m.group(4).split("//")[0].strip()
        if not X18_RE.search(ops):
            continue
        addr, word, mn = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
        if not in_code(addr):
            stats["x18_outside_fde"] += 1  # data decoded as an instruction: never patched
            continue
        pending.append((addr, word, mn, re.sub(r"\s+", " ", ops)))
    if proc.wait() != 0:
        raise Refuse("llvm-objdump failed")

    if stats["x18_outside_fde"]:
        # Either data that decodes as an x18 instruction or hand-written code
        # without CFI: both need a human decision, never a silent skip.
        raise Refuse(f"{stats['x18_outside_fde']} x18 encodings lie outside every FDE")
    mc = os.path.join(llvm_bin, "llvm-mc")
    olds = mc_disasm(mc, [p[1] for p in pending])
    tries, plans = [], []
    for (addr, word, mn, ops), old in zip(pending, olds):
        if old is None or not X18_RE.search(old):
            raise Refuse(f"llvm-objdump/llvm-mc disagree at {addr:#x}")
        named = {int(n) for n in GPR_RE.findall(old)}
        free = [r for r in CANDIDATES if r not in named]
        if len(free) < 2:
            raise Refuse(f"no scratch registers at {addr:#x}: {old}")
        r, r2 = free[0], free[1]
        want = X18_RE.sub(lambda g: f"{g.group(1)}{r}", old)
        fs = [f for f in FIELDS if (word >> f) & 31 == 18]
        subs = []
        for n in range(len(fs), 0, -1):
            for sub in itertools.combinations(fs, n):
                new = word
                for f in sub:
                    new = (new & ~(31 << f)) | (r << f)
                subs.append(new)
                tries.append(new)
        plans.append((addr, word, mn, ops, r, r2, want, subs))
    news = iter(mc_disasm(mc, tries))
    for addr, word, mn, ops, r, r2, want, subs in plans:
        chosen = None
        for new in subs:
            if next(news) == want and chosen is None:
                chosen = new
        kind = x18_kind(mn, ops)
        if chosen is None or kind is None:
            raise Refuse(f"unsupported x18 instruction at {addr:#x}: {mn} {ops}")
        rows.append((addr, f"x18:{kind}", word, chosen, r, r2))
        stats["x18"] += 1

    for name, s in exec_secs:
        blob = data[s["off"]:s["off"] + s["size"]]
        # mrs xN, tpidr_el0 = 0xd53bd040 | N
        for m in re.finditer(rb"[\x40-\x5f]\xd0\x3b\xd5", blob):
            if m.start() % 4:
                continue
            addr = s["va"] + m.start()
            word = struct.unpack_from("<I", blob, m.start())[0]
            n = word & 31
            # Every aligned match is listed, with or without an FDE: a stray
            # match in data fails closed in the loader (text-segment check).
            if n in (18, 31):
                raise Refuse(f"TLS read into register {n} at {addr:#x} has no island shape")
            rows.append((addr, "tls", word, 0, n, 0))
            stats["tls"] += 1
        # svc #imm16 = 0xd4000001 | imm16 << 5
        for m in re.finditer(rb"[\x01\x21\x41\x61\x81\xa1\xc1\xe1][\x00-\xff][\x00-\x1f]\xd4", blob):
            if m.start() % 4:
                continue
            rows.append((s["va"] + m.start(), "svc", struct.unpack_from("<I", blob, m.start())[0], 0, 0, 0))
            stats["svc"] += 1
        # any other system-register access is recorded as a count only
        for m in re.finditer(rb"[\x00-\xff]{2}[\x10-\x1f\x30-\x3f]\xd5", blob):
            if m.start() % 4 == 0 and in_code(s["va"] + m.start()):
                w = struct.unpack_from("<I", blob, m.start())[0]
                if (w >> 5) & 0x7FFF != 0x5E82:
                    stats["other_mrs"] += 1
        # ldr xN, <literal> whose 8-byte literal is an image address
        for m in re.finditer(rb"[\x00-\xff]{3}\x58", blob):
            if m.start() % 4:
                continue
            addr = s["va"] + m.start()
            # No FDE requirement: the only users are glibc's hand-written
            # crt1 `_start` (no CFI). A data word that merely decodes as a
            # literal load is filtered by the in-image test on its literal.
            w = struct.unpack_from("<I", blob, m.start())[0]
            imm = (w >> 5) & 0x7FFFF
            imm -= (imm >> 18) << 19
            target = addr + 4 * imm
            if not (s["va"] <= target and target + 8 <= s["va"] + s["size"]):
                continue
            value = struct.unpack_from("<Q", blob, target - s["va"])[0]
            if not shared and lo <= value < hi:
                rows.append((target, "abs64", value & 0xFFFFFFFF, value >> 32, 0, 0))
                stats["abs64"] += 1

    function_starts = {start for start, _end in fdes}
    function_starts.update(secs[n]["va"] for n in (".init", ".fini") if n in secs)
    alloc = [(n, s) for n, s in secs.items() if s["flags"] & 2 and s["size"] and n != ".tbss"]

    def contradicted(v):
        """True when an in-range value cannot be an address the binary uses."""
        home = next(((n, s) for n, s in alloc if s["va"] <= v < s["va"] + s["size"]), None)
        if home is None or home[0] in UNWIND_SECTIONS:
            return True
        if home[1]["flags"] & 4:
            in_plt = home[0] == ".plt" and (v - plt["va"]) % 16 == 0
            return not (v in function_starts or in_plt)
        return False

    rebase = []
    for name in (() if shared else REBASE_SECTIONS):
        s = secs[name]
        skip = (-s["va"]) % 8
        count, h = 0, hashlib.sha256()
        for k in range((s["size"] - skip) // 8):
            (v,) = struct.unpack_from("<Q", data, s["off"] + skip + 8 * k)
            if lo <= v < hi:
                addr = s["va"] + skip + 8 * k
                if name not in POINTER_ONLY_SECTIONS and contradicted(v):
                    rows.append((addr, "norebase", v & 0xFFFFFFFF, v >> 32, 0, 0))
                    stats["norebase"] += 1
                    continue
                count += 1
                h.update(struct.pack("<Q", addr))
        rebase.append((name, s["va"], s["size"], count, h.hexdigest()))

    rows = sorted(set(rows))
    out = [f"# {SCHEMA}", f"# elf_sha256={sha}", f"# image_lo={lo:#x} image_hi={hi:#x}",
           "# counts " + " ".join(f"{k}={stats[k]}" for k in sorted(stats))]
    for name, va, size, count, digest in rebase:
        out.append(f"# rebase {name} {va:#x} {size:#x} {count} {digest}")
    out.append("# addr\tkind\told_word\tnew_word\tr\tr2")
    for addr, kind, old, new, r, r2 in rows:
        out.append(f"{addr:#x}\t{kind}\t{old:#010x}\t{new:#010x}\tx{r}\tx{r2}")
    return "\n".join(out) + "\n", stats, rebase


def main():
    ap = argparse.ArgumentParser(description=__doc__ if __doc__ else "tileiras patch manifest")
    ap.add_argument("--elf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llvm-bin", default="/opt/homebrew/opt/llvm@21/bin")
    a = ap.parse_args()
    try:
        text, stats, rebase = build(a.elf, a.llvm_bin)
    except Refuse as e:
        print(f"gen_patch_manifest: REFUSED: {e}", file=sys.stderr)
        return 2
    tmp = f"{a.out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, a.out)
    print("counts " + " ".join(f"{k}={stats[k]}" for k in sorted(stats)))
    for name, _va, _size, count, _digest in rebase:
        print(f"rebase {name} {count}")
    print(f"manifest_sha256={hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
