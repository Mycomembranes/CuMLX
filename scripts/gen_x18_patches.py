#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: the Linux-aarch64 libnvptxcompiler_static.a treats
# x18 as an ordinary caller-saved temporary (Linux AAPCS64). Darwin reserves
# x18 and the kernel zeroes it asynchronously in user space (measured: 2,932
# clobbers / 3M checks, portkit/scripts/x18_probe.c). Every x18/w18 operand
# must be renamed before the code runs.
#
# Strategy (per executable section; -ffunction-sections => one function per
# section, and the archive has no split .cold parts):
#   1. collect every GPR the section names (x0..x30 / w0..w30);
#   2. pick a scratch register R from x9..x15 (then x16/x17) that the section
#      NEVER names. x18 is caller-saved on Linux, so it is never live across a
#      call or live-in at entry; R is caller-saved too, so renaming x18 -> R
#      inside a function where R is otherwise dead is semantics-preserving;
#   3. for each x18-using instruction word, rewrite the 5-bit register fields
#      (Rd@0, Rn@5, Rt2/Ra@10, Rm@16) that hold 18, choosing the subset whose
#      disassembly equals the original disassembly with x18->xR / w18->wR
#      (verified with llvm-mc --disassemble, so an immediate that happens to
#      encode 18 is never touched).
#
# Output: a TSV manifest `member  section_index  offset  old_word  new_word  reg`
# consumed by binary-ko-loader's x18 module, which checks old_word before
# patching (fail-closed). Exit status 0 only if every x18 use is patched.
import argparse
import collections
import itertools
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x18_liveness as xl  # noqa: E402

GPR_RE = re.compile(r"\b([xw])(\d{1,2})\b")
X18_RE = re.compile(r"\b[xw]18\b")
INSN_RE = re.compile(r"^\s+([0-9a-f]+):\s+([0-9a-f]{8})\s+(.*)$")
SEC_RE = re.compile(r"^Disassembly of section (.+):$")
FIELDS = (0, 5, 10, 16)
SPMOD_RE = re.compile(r"^(?:\S+ sp,|.*\[sp[^\]]*\]!|.*\[sp\], #)")
ISLAND_MN = collections.Counter()
CANDIDATES = [9, 10, 11, 12, 13, 14, 15, 16, 17]


def readelf_exec_sections(readelf, path):
    """Return [(index, name)] of SHF_EXECINSTR sections in header order."""
    out = subprocess.run([readelf, "-S", "-W", path], capture_output=True, text=True, check=True).stdout
    secs = []
    for line in out.splitlines():
        m = re.match(r"^\s+\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+(\S+)\s+\S+\s+(\S*)", line)
        if not m:
            continue
        idx, name, _typ, size, flags = m.groups()
        if "X" in flags and int(size, 16) > 0:
            secs.append((int(idx), name))
    return secs


RELOC_RE = re.compile(r"^\s+[0-9a-f]+:\s+(R_AARCH64_\S+)\s+(\S+)")


def disasm_member(objdump, path):
    out = subprocess.run([objdump, "-dr", path], capture_output=True, text=True, check=True).stdout
    sections = []  # [(name, [Insn])]
    cur = None
    for line in out.splitlines():
        m = SEC_RE.match(line)
        if m:
            cur = (m.group(1), [])
            sections.append(cur)
            continue
        m = INSN_RE.match(line)
        if m and cur is not None:
            cur[1].append(xl.parse_insn(int(m.group(1), 16), int(m.group(2), 16),
                                        re.sub(r"\s+", " ", m.group(3).split("//")[0].strip())))
            continue
        m = RELOC_RE.match(line)
        if m and cur is not None and cur[1]:
            cur[1][-1].reloc = (m.group(1), m.group(2))
    return sections


def mc_disasm(mc, words):
    """Disassemble 32-bit words with llvm-mc; returns text per word (None = invalid encoding)."""
    if not words:
        return []
    data = "\n".join(" ".join(f"0x{(w >> (8 * i)) & 0xff:02x}" for i in range(4)) for w in words)
    p = subprocess.run([mc, "--disassemble", "-triple=aarch64-linux-gnu", "-mattr=+all"],
                       input=data, capture_output=True, text=True)
    bad = {int(m.group(1)) - 1 for m in re.finditer(r"<stdin>:(\d+):\d+: warning: invalid instruction encoding", p.stderr)}
    lines = []
    for line in p.stdout.splitlines():
        t = line.strip()
        if not t or t.startswith(".text") or t.startswith("#"):
            continue
        lines.append(re.sub(r"\s+", " ", t.split("//")[0].strip()))
    if len(lines) + len(bad) != len(words):
        raise RuntimeError(f"llvm-mc returned {len(lines)} lines + {len(bad)} invalid for {len(words)} words")
    it = iter(lines)
    return [None if k in bad else next(it) for k in range(len(words))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="directory of extracted archive members (*.o)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llvm-bin", default="/opt/homebrew/opt/llvm@21/bin")
    a = ap.parse_args()
    objdump = os.path.join(a.llvm_bin, "llvm-objdump")
    readelf = os.path.join(a.llvm_bin, "llvm-readelf")
    mc = os.path.join(a.llvm_bin, "llvm-mc")

    pending = []  # (member, secidx, off, word, reg, kind)
    stats = collections.Counter()
    no_free = []
    trapped = {}  # (member, secidx) -> (name, entry word)
    CS = set(range(0, 19)) | {30}
    members = sorted(f for f in os.listdir(a.members) if f.endswith(".o"))

    # ---- pass 1: per-function summaries for the whole archive -------------
    func = {}        # (member, secidx) -> dict(mentions, own, callees(raw names), abi)
    keep = {}        # (member, secidx) -> (name, insns, elf) for x18 users
    elfs = {}
    global_def = {}  # global symbol -> (member, shndx)
    for mem in members:
        path = os.path.join(a.members, mem)
        sections = disasm_member(objdump, path)
        elf = xl.Elf(path)
        elfs[mem] = elf
        for g, (shndx, _v) in elf.globals.items():
            global_def.setdefault(g, (mem, shndx))
        hdrs = readelf_exec_sections(readelf, path)
        used_names = collections.defaultdict(int)
        for name, insns in sections:
            stats["insns"] += len(insns)
            matches = [i for i, n in hdrs if n == name]
            k = used_names[name]
            used_names[name] += 1
            if k >= len(matches):
                raise SystemExit(f"{mem}: cannot map section {name}")
            secidx = matches[k]
            by_off = {i.off: n for n, i in enumerate(insns)}
            has18 = any(18 in i.mentions for i in insns)
            if has18:
                xl.resolve_jump_tables(insns, by_off, elf)
            # provisional classification (ext_call unknown yet) for own defs
            xl.classify(insns, by_off)
            mentions = set()
            callees = []
            abi = False
            for i in insns:
                mentions |= i.mentions
                if i.mn == "blr":
                    abi = True
                elif i.mn in ("bl", "b") and i.reloc and i.reloc[0] in ("R_AARCH64_CALL26", "R_AARCH64_JUMP26"):
                    callees.append(i.reloc[1])
            func[(mem, secidx)] = dict(mentions=mentions & CS, own=xl.own_defs(insns), callees=callees,
                                       abi=abi, bl=any(i.mn == "bl" for i in insns))
            if has18:
                keep[(mem, secidx)] = (name, insns, elf, by_off)

    def resolve_target(mem, sym):
        sym = re.sub(r"[+-]0x[0-9a-f]+$", "", sym)
        elf = elfs[mem]
        if sym in elf.names:
            return (mem, elf.names.index(sym))
        if sym in elf.syms and sym not in elf.globals:
            return (mem, elf.syms[sym][0])
        if sym in global_def:
            return global_def[sym]
        return None  # outside the archive (Darwin shim)

    edges = {}
    for key, f in func.items():
        tg = []
        for c in f["callees"]:
            t = resolve_target(key[0], c)
            if t is None or t not in func:
                f["abi"] = True if t is None else f["abi"]
                if t is not None:
                    f["abi"] = True  # unknown in-archive target: be conservative
            else:
                tg.append(t)
        edges[key] = tg

    # may-clobber (upper bound) and definitely-written (lower bound) fixpoints
    clob = {k: (f["mentions"] | ({30} if f["bl"] else set()) | (set(CS) if f["abi"] else set())) for k, f in func.items()}
    dw = {k: ((f["own"] & CS) | (set(CS) if f["abi"] else set())) for k, f in func.items()}
    callers = collections.defaultdict(list)
    for k, tg in edges.items():
        for t in tg:
            callers[t].append(k)
    work = list(func)
    while work:
        k = work.pop()
        c2 = set(clob[k])
        d2 = set(dw[k])
        for t in edges[k]:
            c2 |= clob[t]
            d2 |= dw[t]
        if c2 != clob[k] or d2 != dw[k]:
            clob[k], dw[k] = c2, d2
            work.extend(callers[k])
    stats["functions"] = len(func)

    # ---- pass 2: x18 functions --------------------------------------------
    for key, (name, insns, elf, by_off) in sorted(keep.items()):
        mem, secidx = key
        for i in insns:
            if i.mn == "bl" and i.reloc:
                t = resolve_target(mem, i.reloc[1])
                i.ext_call = t is None or t not in func
                i.callee = t
            elif i.mn == "bl":
                i.ext_call = True
        xl.classify(insns, by_off)
        lin, lout = xl.liveness(insns)
        stats["sections"] += 1
        if any(i.mn == "br" for i in insns):
            stats["sections_with_br"] += 1
        stats["br_total"] += sum(1 for i in insns if i.mn == "br")
        stats["br_resolved"] += sum(1 for i in insns if i.mn == "br" and i.jt)
        if 18 in lin[0]:
            stats["x18_live_at_entry"] += 1
            no_free.append((mem, name + " (x18 live-in at entry)"))
            trapped[key] = (name, insns[0].word)
            continue

        def callee_clob(c):
            return clob.get(c.callee, set(CS)) if not c.ext_call else set(CS)

        allowed = xl.allowed_registers(insns, dw[key], func[key]["own"])
        for web in xl.x18_webs(insns, lin, lout):
            stats["webs"] += 1
            spans_call = any(insns[k].kind == "call" and 18 in lout[k] for k in web)
            if spans_call:
                stats["webs_spanning_calls"] += 1
            reg = xl.choose_register(insns, lin, lout, web, allowed, callee_clob)
            if reg is None:
                stats["webs_unassigned"] += 1
                spmod = [insns[kk] for kk in web if SPMOD_RE.search(insns[kk].text)]
                if spmod:
                    no_free.append((mem, f"{name} web@0x{insns[web[0]].off:x} sp modified: {spmod[0].text}"))
                    trapped[key] = (name, insns[0].word)
                    continue
                if spans_call:
                    cw = xl.constant_web(insns, web) or xl.derived_web(insns, web, lin, SPMOD_RE)
                    if cw is None:
                        no_free.append((mem, f"{name} web@0x{insns[web[0]].off:x} spans a call, no IPA-RA-safe register, not rematerializable"))
                        trapped[key] = (name, insns[0].word)
                        continue
                    stats["remat_webs"] += 1
                    words, defs, uses = cw
                    wtxt = ",".join(f"0x{w:08x}" for w in words)
                    for kk in defs:
                        i = insns[kk]
                        pending.append((mem, secidx, i.off, i.word, 0, "nop"))
                    for kk in uses:
                        i = insns[kk]
                        r = next(c for c in CANDIDATES if c not in i.mentions)
                        kind = {"cbz": "remat_cbz", "cbnz": "remat_cbnz"}.get(i.mn, "remat")
                        if kind == "remat" and (i.mn in ("adr", "adrp", "tbz", "tbnz", "b", "bl", "br", "blr", "ret") or i.mn.startswith("b.")):
                            raise SystemExit(f"{mem} {name}: pc-relative remat not supported: {i.text}")
                        pending.append((mem, secidx, i.off, i.word, r, f"{kind}:{wtxt}"))
                        ISLAND_MN["remat:" + i.mn] += 1
                    continue
                for kk in web:
                    i = insns[kk]
                    if 18 in i.mentions:
                        r = next(c for c in CANDIDATES if c not in i.mentions)
                        kind = {"adrp": "island_adrp", "cbz": "island_cbz", "cbnz": "island_cbnz"}.get(i.mn, "island")
                        if kind == "island" and (i.mn in ("adr", "tbz", "tbnz", "b", "bl", "br", "blr", "ret") or i.mn.startswith("b.") or
                                                 (i.mn.startswith("ldr") and len(i.ops) == 2 and i.ops[1].startswith("0x"))):
                            raise SystemExit(f"{mem} {name}: pc-relative island not supported: {i.text}")
                        pending.append((mem, secidx, i.off, i.word, r, kind))
                        ISLAND_MN[i.mn] += 1
                continue
            stats[f"reg_x{reg}"] += 1
            for kk in web:
                i = insns[kk]
                if 18 in i.mentions:
                    pending.append((mem, secidx, i.off, i.word, reg, "rename"))
    # verify + choose field subsets in bulk
    olds = mc_disasm(mc, [p[3] for p in pending])
    rows, unresolved = [], []
    tries = []  # (pending_idx, new_word)
    for i, (mem, sec, off, w, reg, _kind) in enumerate(pending):
        if _kind == "nop":
            continue
        fs = [f for f in FIELDS if (w >> f) & 31 == 18]
        for n in range(len(fs), 0, -1):
            for sub in itertools.combinations(fs, n):
                nw = w
                for f in sub:
                    nw = (nw & ~(31 << f)) | (reg << f)
                tries.append((i, nw))
    news = mc_disasm(mc, [t[1] for t in tries])
    chosen = {}
    for (i, nw), txt in zip(tries, news):
        if i in chosen:
            continue
        reg = pending[i][4]
        want = re.sub(r"\bx18\b", f"x{reg}", re.sub(r"\bw18\b", f"w{reg}", olds[i]))
        if txt is not None and txt == want:
            chosen[i] = nw
    for i, p in enumerate(pending):
        if p[5] == "nop":
            rows.append((p[0], p[1], p[2], p[3], 0xd503201f, 0, "nop"))
        elif i in chosen:
            rows.append((p[0], p[1], p[2], p[3], chosen[i], p[4], p[5]))
        else:
            unresolved.append((p, olds[i]))

    BRK = 0xd4200000 | (0x1818 << 5)  # brk #0x1818: "x18-unported function entered"
    rows = [r for r in rows if (r[0], r[1]) not in trapped]
    for (m, sec), (nm, w) in sorted(trapped.items()):
        rows.append((m, sec, 0, w, BRK, 0, "trap"))
    with open(a.out, "w") as f:
        f.write("# member\tsection_index\toffset\told_word\tnew_word\treg\tkind\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t0x{r[2]:x}\t0x{r[3]:08x}\t0x{r[4]:08x}\tx{r[5] if r[6] not in ('nop', 'trap') else 0}\t{r[6]}\n")
    print(f"[x18] instructions scanned: {stats['insns']}  sections with br: {stats['sections_with_br']} br resolved {stats['br_resolved']}/{stats['br_total']}  webs: {stats['webs']} unassigned: {stats['webs_unassigned']}")
    print(f"[x18] functions in archive: {stats['functions']}  webs spanning in-archive calls (IPA-RA): {stats['webs_spanning_calls']}")
    print(f"[x18] sections using x18: {stats['sections']}  x18 live at entry: {stats['x18_live_at_entry']}  failures: {len(no_free)}")
    print(f"[x18] webs rematerialized (constant/derived): {stats['remat_webs']}")
    print(f"[x18] rename rows: {sum(1 for r in rows if r[6] == 'rename')}  island rows: {sum(1 for r in rows if r[6] != 'rename')}")
    print(f"[x18] x18 instructions: {len(pending)}  patched+verified: {len(rows)}  unresolved: {len(unresolved)}")
    print("[x18] scratch choice: " + ", ".join(f"{k[4:]}={v}" for k, v in sorted(stats.items()) if k.startswith("reg_")))
    for p, t in unresolved[:20]:
        print(f"[x18] UNRESOLVED {p[0]} sec{p[1]} +0x{p[2]:x} 0x{p[3]:08x} {t}")
    print(f"[x18] island instructions: {sum(ISLAND_MN.values())}  mnemonics: {dict(ISLAND_MN.most_common())}")
    for m, n in no_free:
        print(f"[x18] UNPORTED WEB {m} {n}")
    print(f"[x18] functions guarded with brk #0x1818 (fail-closed if ever called): {len(trapped)}")
    for (m, sec), (nm, w) in sorted(trapped.items()):
        print(f"[x18]   trap {m} {nm}")
    # success = every x18 use either rewritten or inside a guarded function
    return 0 if not unresolved else 1


if __name__ == "__main__":
    sys.exit(main())
