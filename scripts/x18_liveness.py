# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: conservative GPR liveness over one AArch64 function
# (one -ffunction-sections section) as printed by `llvm-objdump -dr`. Used by
# gen_x18_patches.py to find a register that is dead AND untouched over every
# point where x18 is live, so x18 can be renamed to it.
#
# Soundness rule: every approximation OVER-estimates liveness.
#   * every GPR mentioned by an instruction counts as a USE;
#   * a mention counts as a DEF (kills liveness) only for operand positions that
#     are certainly pure destinations (op0 of ordinary data-processing / loads,
#     op0+op1 of ldp-family). Read-modify-write (movk/bfi/bfxil/...) are use+def;
#   * bl/blr: uses x0..x7 (+ blr target), defs x0..x18 + x30 (AAPCS64 clobbers);
#   * ret / tail call (b with a relocation) / br without a resolved table: uses
#     x0..x8 + x19..x30 (ABI live-out). An unresolved br's successors are every
#     instruction of the function (jump-table targets stay inside the function);
#   * brk/udf/hlt: no successors.
import re

GPR_RE = re.compile(r"\b([xw])(\d{1,2})\b")
NO_DEF = {
    "cmp", "cmn", "tst", "ccmp", "ccmn", "prfm", "prfum", "b", "br", "blr", "bl", "ret",
    "cbz", "cbnz", "tbz", "tbnz", "nop", "hint", "dmb", "dsb", "isb", "msr", "sys", "dc",
    "ic", "brk", "udf", "hlt", "bti", "paciasp", "autiasp", "yield", "clrex", "fcmp", "fcmpe",
    "fccmp", "fccmpe",
}
RMW = {"movk", "bfi", "bfxil", "bfm", "bfc"}
PAIR_DEF = {"ldp", "ldpsw", "ldnp", "ldxp", "ldaxp"}
ABI_OUT = set(range(0, 9)) | set(range(19, 31))
CALL_CLOBBER = set(range(0, 19)) | {30}
CALL_USE = set(range(0, 8))
ALL = set(range(0, 31))


def gprs(tok):
    m = re.fullmatch(r"([xw])(\d{1,2})", tok.strip())
    if m and int(m.group(2)) <= 30:
        return int(m.group(2))
    return None


def split_ops(ops):
    out, depth, cur = [], 0, ""
    for ch in ops:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Insn:
    __slots__ = ("off", "word", "mn", "ops", "text", "reloc", "uses", "defs", "mentions", "succ", "kind", "target", "jt", "ext_call", "callee")


def parse_insn(off, word, text):
    i = Insn()
    i.off, i.word, i.text, i.reloc = off, word, text, None
    parts = text.split(None, 1)
    i.mn = parts[0]
    ops = parts[1] if len(parts) > 1 else ""
    ops = ops.split("//")[0].strip()
    i.ops = split_ops(ops)
    i.mentions = {int(n) for _, n in GPR_RE.findall(ops) if int(n) <= 30}
    i.target = None
    i.jt = None
    i.ext_call = False
    i.callee = None
    m = re.match(r"^(0x[0-9a-f]+)\b", i.ops[-1]) if i.ops else None
    if i.mn in ("b", "bl") or i.mn.startswith("b.") or i.mn in ("cbz", "cbnz", "tbz", "tbnz"):
        if m:
            i.target = int(m.group(1), 16)
    return i


def classify(insns, by_off):
    """Fill uses/defs/succ for every instruction (needs .reloc already set)."""
    n = len(insns)
    for k, i in enumerate(insns):
        mn = i.mn
        i.uses = set(i.mentions)
        i.defs = set()
        nxt = [k + 1] if k + 1 < n else []
        if mn == "bl":
            i.uses |= CALL_USE
            # GCC IPA-RA: a caller may keep ANY register (x18 included) live across
            # a call to a function it knows does not clobber it. Only calls to
            # code outside the archive (Darwin shims via veneers) are guaranteed
            # to follow the AAPCS64 clobber set; in-archive calls kill only x30.
            i.defs = set(CALL_CLOBBER) if getattr(i, "ext_call", False) else {30}
            i.succ = nxt
            i.kind = "call"
        elif mn == "blr":
            i.uses |= CALL_USE
            i.defs = set(CALL_CLOBBER)
            i.succ = nxt
            i.kind = "call"
        elif mn == "ret":
            i.uses |= ABI_OUT
            i.succ = []
            i.kind = "ret"
        elif mn == "br":
            if getattr(i, "jt", None):
                i.succ = list(i.jt)  # decoded jump table
                i.kind = "brjt"
            else:
                i.uses |= ABI_OUT
                i.succ = list(range(n))  # unresolved: any instruction of the function
                i.kind = "br"
        elif mn in ("brk", "udf", "hlt"):
            i.succ = []
            i.kind = "trap"
        elif mn == "b":
            if i.reloc or i.target not in by_off:
                i.uses |= ABI_OUT
                i.succ = []
                i.kind = "tail"
            else:
                i.succ = [by_off[i.target]]
                i.kind = "jmp"
        elif mn.startswith("b.") or mn in ("cbz", "cbnz", "tbz", "tbnz"):
            if i.reloc or i.target not in by_off:
                i.uses |= ABI_OUT
                i.succ = nxt
            else:
                i.succ = nxt + [by_off[i.target]]
            i.kind = "cond"
        else:
            i.succ = nxt
            i.kind = "op"
            if mn in NO_DEF or mn.startswith("st") or mn.startswith("cas") or mn.startswith("swp") \
                    or mn.startswith("ldadd") or mn.startswith("ldclr") or mn.startswith("ldset") \
                    or mn.startswith("ldeor") or mn.startswith("ldsmax") or mn.startswith("ldsmin") \
                    or mn.startswith("ldumax") or mn.startswith("ldumin"):
                pass
            elif mn in PAIR_DEF:
                addr = set()
                for op in i.ops[2:]:
                    addr |= {int(x) for _, x in GPR_RE.findall(op) if int(x) <= 30}
                for op in i.ops[:2]:
                    r = gprs(op)
                    if r is not None:
                        i.defs.add(r)
                        if r not in addr:
                            i.uses.discard(r)
            elif i.ops:
                r = gprs(i.ops[0])
                if r is not None:
                    i.defs.add(r)
                    if mn in RMW:
                        pass  # stays in uses too (mentions)
                    else:
                        # op0 is a pure destination; it is still a use only if it is
                        # also mentioned in another operand.
                        others = set()
                        for op in i.ops[1:]:
                            others |= {int(x) for _, x in GPR_RE.findall(op) if int(x) <= 30}
                        if r not in others:
                            i.uses.discard(r)


def liveness(insns):
    n = len(insns)
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    preds = [[] for _ in range(n)]
    for k, i in enumerate(insns):
        for s in i.succ:
            preds[s].append(k)
    work = list(range(n - 1, -1, -1))
    inq = [True] * n
    while work:
        k = work.pop()
        inq[k] = False
        i = insns[k]
        out = set()
        for s in i.succ:
            out |= live_in[s]
        new_in = i.uses | (out - i.defs)
        live_out[k] = out
        if new_in != live_in[k]:
            live_in[k] = new_in
            for p in preds[k]:
                if not inq[p]:
                    inq[p] = True
                    work.append(p)
    return live_in, live_out


def x18_webs(insns, live_in, live_out):
    """Connected components of the instruction points where x18 is live or mentioned."""
    pts = [k for k, i in enumerate(insns) if 18 in i.mentions or 18 in live_in[k] or 18 in live_out[k]]
    ptset = set(pts)
    parent = {p: p for p in pts}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in pts:
        if 18 in live_out[k]:
            for s in insns[k].succ:
                if s in ptset and 18 in live_in[s]:
                    parent[find(k)] = find(s)
    webs = {}
    for p in pts:
        webs.setdefault(find(p), []).append(p)
    return list(webs.values())


def choose_register(insns, live_in, live_out, web, allowed, callee_clob):
    """First allowed register dead+untouched over the web and not clobbered by
    any in-archive callee the web spans (callee_clob(insn) -> may-clobber set)."""
    spanned = [insns[k] for k in web if insns[k].kind == "call" and 18 in live_out[k]]
    for r in allowed:
        ok = True
        for k in web:
            if r in insns[k].mentions or r in live_in[k] or r in live_out[k]:
                ok = False
                break
        if ok:
            for c in spanned:
                if r in callee_clob(c):
                    ok = False
                    break
        if ok:
            return r
    return None


def allowed_registers(insns, definitely_written, own):
    """Scratch candidates that cannot break a caller relying on IPA-RA:
    caller-saved registers this function already (definitely) clobbers, then
    callee-saved registers it already saves/writes. Never x29/x30/x18."""
    cs = [r for r in (9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8, 0) if r in definitely_written]
    ce = [r for r in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28) if r in own]
    return cs + ce


def own_defs(insns):
    """Registers definitely written by non-call instructions of a function."""
    d = set()
    for i in insns:
        if i.kind != "call":
            d |= i.defs
    return d


# ---------------------------------------------------------------------------
# Jump tables (GCC aarch64 pattern):
#   cmp  wI, #N ; b.hi/b.ls ...            -> N+1 entries
#   adrp xT, TABLE ; add xT, xT, :lo12:TABLE   (relocations name TABLE)
#   ldr{b,h,}  wI, [xT, wI, uxtw #s]       -> entry size 1<<s, signed
#   adr  xB, BASE ; add xD, xB, wI, sxt{b,h,w} #2 ; br xD
# target = BASE + entry*4
# ---------------------------------------------------------------------------
import struct  # noqa: E402


class Elf:
    def __init__(self, path):
        b = open(path, "rb").read()
        self.b = b
        shoff, = struct.unpack_from("<Q", b, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
        self.sh = []
        for k in range(shnum):
            name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
                "<IIQQQQIIQQ", b, shoff + k * shentsize)
            self.sh.append((name, typ, off, size, link))
        stroff = self.sh[shstrndx][2]
        self.names = [self._str(stroff, s[0]) for s in self.sh]
        self.syms = {}
        self.globals = {}
        for k, s in enumerate(self.sh):
            if s[1] == 2:  # SYMTAB
                strtab = self.sh[s[4]][2]
                for j in range(s[3] // 24):
                    nm, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", b, s[2] + j * 24)
                    n = self._str(strtab, nm)
                    if n and shndx < len(self.sh):
                        self.syms.setdefault(n, (shndx, value))
                        if (info >> 4) != 0 and shndx != 0:
                            self.globals.setdefault(n, (shndx, value))

    def _str(self, off, idx):
        e = self.b.index(b"\0", off + idx)
        return self.b[off + idx:e].decode("latin1")

    def resolve(self, symexpr):
        m = re.match(r"^(.*?)(?:([+-])0x([0-9a-f]+))?$", symexpr)
        name, sign, add = m.group(1), m.group(2), m.group(3)
        addend = int(add, 16) * (-1 if sign == "-" else 1) if add else 0
        if name in self.names:
            idx = self.names.index(name)
            return idx, addend
        if name in self.syms:
            shndx, value = self.syms[name]
            return shndx, value + addend
        return None

    def read(self, shndx, off, size):
        s = self.sh[shndx]
        if off < 0 or off + size > s[3]:
            return None
        return self.b[s[2] + off:s[2] + off + size]


FLAG_SETTERS = {"cmp", "cmn", "tst", "adds", "subs", "ands", "bics", "adcs", "sbcs", "ccmp", "ccmn",
                "negs", "ngcs", "fcmp", "fcmpe", "fccmp", "fccmpe", "msr", "setf8", "setf16", "rmif"}


def _is_branch(i):
    return i.mn in ("b", "bl", "br", "blr", "ret", "cbz", "cbnz", "tbz", "tbnz") or i.mn.startswith("b.")


def _op0_def(i):
    r = gprs(i.ops[0]) if i.ops else None
    if r is None or i.mn.startswith("st") or i.mn in NO_DEF:
        return set()
    return {r}


Insn.kind_hint_branch = _is_branch
Insn.defs_hint = _op0_def


def resolve_jump_tables(insns, by_off, elf):
    """Set i.jt = [target indices] on every br whose table can be decoded."""
    for k, i in enumerate(insns):
        if i.mn != "br":
            continue
        i.jt = None
        try:
            dreg = gprs(i.ops[0])
            a = insns[k - 1]
            if a.mn != "add" or gprs(a.ops[0]) != dreg or len(a.ops) < 4:
                continue
            breg, ireg = gprs(a.ops[1]), gprs(a.ops[2])
            if not re.match(r"sxt([bhw]) #2", a.ops[3]):
                continue
            base = None
            ld = None
            for j in range(k - 2, max(-1, k - 12), -1):
                q = insns[j]
                if base is None and q.mn == "adr" and gprs(q.ops[0]) == breg:
                    base = int(re.match(r"(0x[0-9a-f]+)", q.ops[1]).group(1), 16)
                if ld is None and q.mn in ("ldrb", "ldrh", "ldr", "ldrsb", "ldrsh", "ldrsw") \
                        and gprs(q.ops[0]) == ireg and "uxtw" in q.text:
                    ld = j
                if base is not None and ld is not None:
                    break
            if base is None or ld is None:
                continue
            q = insns[ld]
            esize = {"ldrb": 1, "ldrsb": 1, "ldrh": 2, "ldrsh": 2, "ldr": 4, "ldrsw": 4}[q.mn]
            mm = re.match(r"\[x(\d+), w(\d+), uxtw(?: #(\d))?\]", q.ops[1] if len(q.ops) > 1 else "")
            if not mm:
                continue
            treg, idxreg = int(mm.group(1)), int(mm.group(2))
            tbl = None
            for j in range(ld - 1, max(-1, ld - 12), -1):
                p = insns[j]
                if p.mn == "add" and gprs(p.ops[0]) == treg and p.reloc and p.reloc[0] == "R_AARCH64_ADD_ABS_LO12_NC":
                    tbl = elf.resolve(p.reloc[1])
                    break
            if tbl is None:
                continue
            n = None
            for j in range(ld - 1, max(-1, ld - 24), -1):
                p = insns[j]
                if p.mn == "cmp" and gprs(p.ops[0]) == idxreg and p.ops[1].startswith("#"):
                    # first conditional branch after the cmp, skipping instructions
                    # that neither set flags nor redefine the index register
                    for j2 in range(j + 1, ld):
                        q2 = insns[j2]
                        if q2.mn.startswith("b."):
                            if q2.mn in ("b.hi", "b.ls", "b.hs", "b.lo"):
                                n = int(p.ops[1][1:], 0) + (1 if q2.mn in ("b.hi", "b.ls") else 0)
                            break
                        if q2.mn in FLAG_SETTERS or q2.kind_hint_branch() or idxreg in q2.defs_hint():
                            break
                    break
            if n is None or n > 70000:
                continue
            raw = elf.read(tbl[0], tbl[1], n * esize)
            if raw is None:
                continue
            fmt = {1: "b", 2: "h", 4: "i"}[esize]
            ents = struct.unpack("<%d%s" % (n, fmt), raw)
            tg = []
            for e in ents:
                t = base + 4 * e
                if t not in by_off:
                    raise ValueError("target outside function")
                tg.append(by_off[t])
            i.jt = sorted(set(tg))
        except (ValueError, IndexError, AttributeError, KeyError):
            i.jt = None


def constant_web(insns, web):
    """If every x18 definition in `web` is a mov/movk chain producing one
    constant and every other x18 mention is a pure use, return
    (value, def_indices, use_indices); else None."""
    pts = sorted(web)
    defs = [k for k in pts if 18 in insns[k].defs]
    uses = [k for k in pts if 18 in insns[k].mentions and 18 not in insns[k].defs]
    if not defs or not uses:
        return None
    values = set()
    cur = None
    prev = None
    chains = []
    targets = set()
    for i in insns:
        for t in i.succ:
            targets.add(t)
    for k in defs:
        i = insns[k]
        op0 = i.ops[0] if i.ops else ""
        if i.mn == "mov" and len(i.ops) == 2 and i.ops[1].startswith("#") and op0 in ("x18", "w18"):
            v = int(i.ops[1][1:], 0)
            v &= 0xffffffff if op0 == "w18" else 0xffffffffffffffff
            if cur is not None:
                values.add(cur)
            cur, prev = v, k
            chains.append([i.word])
        elif i.mn == "movk" and op0 == "x18" and cur is not None and prev is not None:
            # must directly follow the previous chain element (straight line)
            if any(insns[j].kind != "op" or 18 in insns[j].mentions or j in targets for j in range(prev + 1, k)):
                return None
            if k in targets:
                return None
            m = re.match(r"#(0x[0-9a-f]+|\d+)(?:, lsl #(\d+))?$", ", ".join(i.ops[1:]))
            if not m:
                return None
            imm, sh = int(m.group(1), 0), int(m.group(2) or 0)
            cur = (cur & ~(0xffff << sh)) | (imm << sh)
            prev = k
            chains[-1].append(i.word)
        else:
            return None
    values.add(cur)
    if len(values) != 1:
        return None
    for k in uses:
        if insns[k].mn in ("movk",):
            return None
    words = chains[0]
    if any((w & 31) != 18 or ((w >> 5) & 31) == 18 or ((w >> 16) & 31) == 18 for w in words):
        return None
    return words, defs, uses


WB_RE = r"\[{b}(?:, [^\]]*)?\]!|\[{b}\], "


def derived_web(insns, web, live_in, spmod_re):
    """x18 defined once as `add/sub xB, #imm` or `mov x18, xB` (B single-def in
    the function and live at every use, or B = sp with no sp change in the web):
    return ([def word], defs, uses)."""
    pts = sorted(web)
    defs = [k for k in pts if 18 in insns[k].defs]
    uses = [k for k in pts if 18 in insns[k].mentions and 18 not in insns[k].defs]
    if len(defs) != 1 or not uses:
        return None
    d = insns[defs[0]]
    if d.mn not in ("add", "sub", "mov") or not d.ops or d.ops[0] != "x18" or len(d.ops) < 2:
        return None
    if d.mn == "mov" and len(d.ops) != 2:
        return None
    if d.mn in ("add", "sub") and (len(d.ops) < 3 or not d.ops[2].startswith("#")):
        return None
    if d.ops[1] == "sp":
        if any(spmod_re.search(insns[k].text) for k in pts):
            return None
        if any(spmod_re.search(i.text) for i in insns[defs[0]:max(uses) + 1]):
            return None
    else:
        b = gprs(d.ops[1])
        if b is None or b == 18 or not d.ops[1].startswith("x"):
            return None
        if any(i.kind == "br" for i in insns):
            return None  # an unresolved jump table makes liveness too coarse here
        # calls are excluded: B live at a use after a call proves the original
        # program relied on the callee preserving B (single definition).
        ndefs = sum(1 for i in insns if i.kind != "call" and b in i.defs)
        wb = re.compile(WB_RE.format(b="x%d" % b))
        if ndefs != 1 or any(wb.search(i.text) for i in insns):
            return None
        if any(b not in live_in[k] for k in uses):
            return None
    w = d.word
    if (w & 31) != 18 or ((w >> 5) & 31) == 18:
        return None
    return [w], defs, uses
