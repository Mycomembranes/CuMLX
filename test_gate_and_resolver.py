#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: host-only tests for solution 105 Route B
#   * scripts/cubin_diff.py `diff` is a GATE (incl. layout normalization + tkinfo descsz)
#     and `check` validates strictly (.nv.info, KPARAM vs PTX) -- on SYNTHETIC cubins
#     built here, so they run on a clean checkout without any NVIDIA bytes;
#   * the same gate on the real reference cubins, and scripts/milestone.sh FAILING on a
#     changed .text byte / corrupt bundle and reporting SKIP (77) without the bundle --
#     these need the external reference bundle and are SKIPPED (with the reason) without it;
#   * tools/ptxas-resolve.sh: the --print-path/--print-identity contract, the external
#     binary pin + private content-addressed copy, never executing a PATH ptxas, and
#     the Linux trusted-root selection (simulated with fake roots + fake ELF headers).
# Stdlib unittest; no GPU, no daemon.
#   python3 -I portkit/ptxas-native/tests/test_gate_and_resolver.py
# PTXAS_NATIVE_REQUIRE_FULL=1 turns every skip into a failure (CI / pre-commit).
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CRATE))
SCRIPTS = os.path.join(CRATE, "scripts")
DIFF = os.path.join(SCRIPTS, "cubin_diff.py")
MILESTONE = os.path.join(SCRIPTS, "milestone.sh")
RESOLVE = os.path.join(REPO, "tools", "ptxas-resolve.sh")
FIXTURES = os.path.join(CRATE, "fixtures")
FX_PTX = os.path.join(FIXTURES, "ptx")
BUNDLE_SHA = os.path.join(FIXTURES, "reference-bundle.sha256")
BUNDLE = os.environ.get("PTXAS_NATIVE_REF_BUNDLE", os.path.join(REPO, "crates", "cuda-driverkit", "test_assets"))
NATIVE_BIN = os.environ.get("PTXAS_DARWIN_ARM64_BIN",
                            os.path.join(CRATE, "target", "release", "ptxas-darwin-arm64"))
IS_MAC_ARM = platform.system() == "Darwin" and platform.machine() == "arm64"
REQUIRE_FULL = os.environ.get("PTXAS_NATIVE_REQUIRE_FULL") == "1"

sys.path.insert(0, SCRIPTS)
import cubin_diff  # noqa: E402


def rd(path, mode="rb"):
    with open(path, mode) as f:
        return f.read()


def wr(path, data, mode="wb"):
    with open(path, mode) as f:
        f.write(data)


def jload(path):
    with open(path) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(rd(path)).hexdigest()


def run(cmd, env=None, **kw):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, capture_output=True, text=True, env=e, **kw)


def require(cond, reason):
    """skipUnless (class or method) that becomes a failure under PTXAS_NATIVE_REQUIRE_FULL=1."""
    if cond:
        return lambda obj: obj
    if not REQUIRE_FULL:
        return unittest.skip(reason)

    def deco(obj):
        def fail(self, *_a, **_k):
            self.fail(f"required prerequisite missing: {reason}")
        if isinstance(obj, type):
            obj.setUp = fail
            return obj
        return fail
    return deco


# ---------------------------------------------------------------------------
# reference bundle (external, NVIDIA-compiled; SHA-pinned)
# ---------------------------------------------------------------------------
def bundle_entries():
    out = {}
    for line in rd(BUNDLE_SHA, "r").splitlines():
        if line and not line.startswith("#"):
            h, rel = line.split(None, 1)
            out[rel.strip()] = h
    return out


def bundle_state():
    """'present' | 'absent' | 'corrupt: <file>'"""
    for rel, h in bundle_entries().items():
        p = os.path.join(BUNDLE, rel)
        if not os.path.isfile(p):
            return "absent"
        if sha(p) != h:
            return f"corrupt: {rel}"
    return "present"


BUNDLE_STATE = bundle_state()
HAVE_BUNDLE = BUNDLE_STATE == "present"
BUNDLE_REASON = (f"reference bundle {BUNDLE_STATE} under {BUNDLE} "
                 "(portkit/ptxas-native/README.md 'Reference bundle')")
REF = os.path.join(BUNDLE, "cubins", "vecadd_macos.cubin")


def native_verified():
    if not IS_MAC_ARM or not os.access(NATIVE_BIN, os.X_OK):
        return False
    return run([NATIVE_BIN, "--identity"]).returncode == 0


NATIVE_OK = native_verified()
NATIVE_REASON = "native ptxas-darwin-arm64 not built/verified (portkit/ptxas-native/build.sh)"


# ---------------------------------------------------------------------------
# synthetic cubin builder (no NVIDIA bytes): enough ELF for every gate/check rule
# ---------------------------------------------------------------------------
def tkinfo(version="Cuda compilation tools, release 13.2, V13.2.51", build="Build system X",
           options="-arch sm_120 ", tool="ptxas"):
    strtab = b"\0" + tool.encode() + b"\0"
    o_ver = len(strtab)
    strtab += version.encode() + b"\0"
    o_build = len(strtab)
    strtab += build.encode() + b"\0"
    o_opts = len(strtab)
    strtab += options.encode() + b"\0"
    desc = struct.pack("<6I", 2, 0, 1, o_ver, o_build, o_opts) + strtab
    desc += b"\0" * (-len(desc) % 4)
    return struct.pack("<III", 12, len(desc), 2000) + b"NVIDIA Corp\0" + desc


def kparam(ordinal, offset, size):
    return bytes([0x04, 0x17, 12, 0]) + struct.pack("<IHHI", 0, ordinal, offset, size << 18)


def nvinfo_k(params=((0, 0, 8), (1, 8, 8), (2, 16, 4)), cbank=20):
    b = b"".join(kparam(*p) for p in params)
    if cbank is not None:
        b += bytes([0x03, 0x19]) + struct.pack("<H", cbank)
    return b


SYN_PTX = ".version 8.7\n.target sm_120\n.address_size 64\n" \
          ".visible .entry k(\n .param .u64 .ptr .align 1 a,\n .param .u64 b, // c\n .param .u32 n\n)\n{ ret; }\n"


def make_cubin(tk=None, text=None, info=None, sm=120, e_type=2, machine=190, gap_after_info=0,
               trailing=b""):
    """ELF64: null, .shstrtab, .note.nv.tkinfo, .nv.info, .nv.info.k, .text.k,
    .nv.constant0.k, then section headers and one PT_LOAD program header."""
    tk = tkinfo() if tk is None else tk
    text = bytes(range(256)) if text is None else text
    info = nvinfo_k() if info is None else info
    names = [".shstrtab", ".note.nv.tkinfo", ".nv.info", ".nv.info.k", ".text.k", ".nv.constant0.k"]
    shstr = b"\0" + b"".join(n.encode() + b"\0" for n in names)
    name_off = {n: shstr.index(b"\0" + n.encode() + b"\0") + 1 for n in names}
    datas = [(".shstrtab", 3, shstr, 1), (".note.nv.tkinfo", 7, tk, 4), (".nv.info", 0x70000000,
             bytes([0x03, 0x1b, 0xff, 0]), 4), (".nv.info.k", 0x70000000, info, 4), (".text.k", 1, text, 128),
             (".nv.constant0.k", 1, bytes(0x380 + 20), 4)]
    body = bytearray(64)
    offs = []
    for name, _ty, data, al in datas:
        body += b"\0" * (-len(body) % al)
        offs.append(len(body))
        body += data
        if name == ".nv.info":
            body += b"\0" * gap_after_info
    body += b"\0" * (-len(body) % 8)
    shoff = len(body)
    sh = bytearray(64)  # null section
    for (name, ty, data, al), off in zip(datas, offs):
        sh += struct.pack("<IIQQQQIIQQ", name_off[name], ty, 6 if name == ".text.k" else 0, 0, off, len(data),
                          0, 0, al, 0)
    body += sh
    phoff = len(body)
    toff = offs[names.index(".text.k")]
    body += struct.pack("<IIQQQQQQ", 1, 5, toff, 0, 0, len(text), len(text), 8)
    body += trailing
    hdr = bytearray(b"\x7fELF\x02\x01\x01\x41\x08" + b"\0" * 7)
    hdr += struct.pack("<HHIQQQIHHHHHH", e_type, machine, 1, 0, phoff, shoff, (sm << 8) | 0x0004,
                       64, 56, 1, 64, len(datas) + 1, 1)
    body[:64] = hdr
    return bytes(body)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptxas-t-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def put(self, name, data, mode="wb"):
        p = os.path.join(self.tmp, name)
        wr(p, data, mode)
        return p


class SyntheticGate(Tmp):
    def gate(self, a, b, *extra):
        return run([sys.executable, "-I", DIFF, "diff", *extra, a, b])

    def pair(self, native, linux):
        return self.put("n.cubin", native), self.put("l.cubin", linux)

    def test_builder_is_a_valid_cubin(self):
        p = self.put("k.cubin", make_cubin())
        ptx = self.put("k.ptx", SYN_PTX, "w")
        r = run([sys.executable, "-I", DIFF, "check", p, "--ptx", ptx])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_identical_passes(self):
        a, b = self.pair(make_cubin(), make_cubin())
        self.assertEqual(self.gate(a, b, "--strict-options").returncode, 0)

    def test_tolerated_tkinfo_with_layout_shift_passes(self):
        native = make_cubin(tkinfo(version="Cuda compilation tools, release 13.2, V13.2.51",
                                   build="Build system must define TOOLS_VERSION_EXTENDED"))
        linux = make_cubin(tkinfo(version="Cuda compilation tools, release 13.2, V13.2.78",
                                  build="Build cuda_13.2.r13.2/compiler.1"))
        a, b = self.pair(native, linux)
        r = self.gate(a, b, "--strict-options")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("tolerated:", r.stdout)
        self.assertNotEqual(cubin_diff.parse(a)["secs"][3]["off"], cubin_diff.parse(b)["secs"][3]["off"],
                            "the test must actually exercise a tkinfo-induced shift")

    def test_text_byte_fails(self):
        t = bytearray(range(256))
        t[17] ^= 1
        a, b = self.pair(make_cubin(text=bytes(t)), make_cubin())
        r = self.gate(a, b)
        self.assertEqual(r.returncode, 1)
        self.assertIn("VIOLATION: .text.k", r.stdout)

    def test_layout_shift_not_from_tkinfo_fails(self):
        a, b = self.pair(make_cubin(gap_after_info=4), make_cubin())
        r = self.gate(a, b)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("layout:", r.stdout)

    def test_trailing_bytes_fail(self):
        a, b = self.pair(make_cubin(trailing=b"\0" * 8), make_cubin())
        self.assertEqual(self.gate(a, b).returncode, 1)

    def test_nonzero_padding_fails(self):
        x = bytearray(make_cubin(gap_after_info=4))
        off = cubin_diff.parse(self.put("x.cubin", bytes(x)))["secs"][3]["off"]
        x[off + 4] = 0x55  # inside the 4-byte gap after .nv.info
        a, b = self.pair(bytes(x), make_cubin(gap_after_info=4))
        r = self.gate(a, b)
        self.assertEqual(r.returncode, 1)
        self.assertIn("outside all sections", r.stdout)

    def test_tkinfo_descsz_padding_mismatch_fails(self):
        tk = bytearray(tkinfo())
        namesz, descsz, typ = struct.unpack_from("<III", tk, 0)
        tk[4:8] = struct.pack("<I", descsz + 4)  # claim 4 more desc bytes ...
        tk += b"\0" * 4                          # ... that are zero padding
        a, b = self.pair(make_cubin(bytes(tk)), make_cubin())
        r = self.gate(a, b)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("tkinfo", r.stdout)

    def test_tkinfo_release_tool_options(self):
        base = make_cubin()
        for tk, strict, want in (
            (tkinfo(version="Cuda compilation tools, release 13.3, V13.3.0"), False, 1),
            (tkinfo(tool="ptxaz"), False, 1),
            (tkinfo(options="-arch sm_121 "), False, 0),
            (tkinfo(options="-arch sm_121 "), True, 1),
        ):
            a, b = self.pair(make_cubin(tk), base)
            args = ["--strict-options"] if strict else []
            self.assertEqual(self.gate(a, b, *args).returncode, want, (tk, strict))

    def test_section_header_flag_fails(self):
        x = bytearray(make_cubin())
        c = cubin_diff.parse(self.put("x.cubin", bytes(x)))
        x[c["shoff"] + 3 * 64 + 8] ^= 1  # sh_flags of .nv.info
        a, b = self.pair(bytes(x), make_cubin())
        r = self.gate(a, b)
        self.assertEqual(r.returncode, 1)
        self.assertIn("sh_flags", r.stdout)

    def test_bad_section_name_offset_is_usage_error(self):
        x = bytearray(make_cubin())
        c = cubin_diff.parse(self.put("x.cubin", bytes(x)))
        x[c["shoff"] + 2 * 64:c["shoff"] + 2 * 64 + 4] = struct.pack("<I", 0xFFFFFF)
        a, b = self.pair(bytes(x), make_cubin())
        r = self.gate(a, b)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("outside .shstrtab", r.stderr)

    def test_truncated_is_usage_error(self):
        a, b = self.pair(make_cubin()[:100], make_cubin())
        self.assertEqual(self.gate(a, b).returncode, 2)


class SyntheticCheck(Tmp):
    def check(self, cubin, ptx=SYN_PTX, sm="120"):
        c = self.put("k.cubin", cubin)
        p = self.put("k.ptx", ptx, "w")
        r = run([sys.executable, "-I", DIFF, "check", c, "--ptx", p, "--expect-sm", sm])
        return r.returncode, json.loads(r.stdout)[0]["problems"] if r.stdout.strip().startswith("[") else r.stderr

    def test_rules(self):
        self.assertEqual(self.check(make_cubin())[0], 0)
        cases = {
            "truncated nv.info record": make_cubin(info=nvinfo_k()[:-6]),
            "unknown nv.info format": make_cubin(info=nvinfo_k() + bytes([0x09, 0x01, 0, 0])),
            "empty .text": make_cubin(text=b""),
            "ordinal gap": make_cubin(info=nvinfo_k(((0, 0, 8), (2, 8, 8), (3, 16, 4)))),
            "overlap": make_cubin(info=nvinfo_k(((0, 0, 8), (1, 4, 8), (2, 16, 4)))),
            "beyond CBANK_PARAM_SIZE": make_cubin(info=nvinfo_k(cbank=12)),
            "no CBANK_PARAM_SIZE": make_cubin(info=nvinfo_k(cbank=None)),
            "size mismatch vs PTX": make_cubin(info=nvinfo_k(((0, 0, 8), (1, 8, 4), (2, 12, 4)), cbank=16)),
            "param count vs PTX": make_cubin(info=nvinfo_k(((0, 0, 8), (1, 8, 8)), cbank=16)),
            "wrong SM": make_cubin(sm=90),
            "ET_REL": make_cubin(e_type=1),
            "not EM_CUDA": make_cubin(machine=183),
        }
        for what, cub in cases.items():
            rc, probs = self.check(cub)
            self.assertNotEqual(rc, 0, f"{what} accepted")
        rc, probs = self.check(make_cubin(), ptx=SYN_PTX.replace("entry k(", "entry other("))
        self.assertNotEqual(rc, 0, "entry-set mismatch accepted")


@require(HAVE_BUNDLE, BUNDLE_REASON)
class RealReferenceGate(Tmp):
    def gate(self, a, b, *extra):
        return run([sys.executable, "-I", DIFF, "diff", *extra, a, b])

    def test_reference_vs_itself_passes_and_text_flip_fails(self):
        self.assertEqual(self.gate(REF, REF, "--strict-options").returncode, 0)
        t = cubin_diff.parse(REF)
        s = next(x for x in t["secs"] if x["name"] == ".text.vecadd_macos")
        b = bytearray(rd(REF))
        b[s["off"] + 16] ^= 1
        r = self.gate(self.put("bad.cubin", bytes(b)), REF)
        self.assertEqual(r.returncode, 1)
        self.assertIn("VIOLATION: .text.vecadd_macos", r.stdout)

    def test_every_reference_tkinfo_parses(self):
        for rel in bundle_entries():
            if rel.endswith(".cubin"):
                c = cubin_diff.parse(os.path.join(BUNDLE, rel))
                tk = next(x for x in c["secs"] if x["name"] == ".note.nv.tkinfo")
                self.assertEqual(cubin_diff.parse_tkinfo(tk["data"])[0]["fields"]["tool"], "ptxas", rel)


@require(NATIVE_OK, NATIVE_REASON)
class Milestone(Tmp):
    def milestone(self, out, bundle):
        return run(["bash", MILESTONE, out], env=dict(
            PTXAS_NATIVE_REF_BUNDLE=bundle, MILESTONE_ONLY="vecadd_macos", MILESTONE_SKIP_NOREF="1",
            MILESTONE_SKIP_INSPECT="1", MILESTONE_SKIP_LOADER_TESTS="1", PTXAS_DARWIN_ARM64=NATIVE_BIN))

    def copy_bundle(self):
        dst = os.path.join(self.tmp, "bundle")
        for rel in bundle_entries():
            os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
            shutil.copy(os.path.join(BUNDLE, rel), os.path.join(dst, rel))
        return dst

    def test_absent_bundle_is_skip_77_not_pass(self):
        r = self.milestone(os.path.join(self.tmp, "o"), os.path.join(self.tmp, "no-bundle"))
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("SKIPPED", r.stdout)
        self.assertIn("MILESTONE: SKIP", r.stdout)
        self.assertEqual(jload(os.path.join(self.tmp, "o", "milestone.json"))["verdict"], "SKIP")

    @require(HAVE_BUNDLE, BUNDLE_REASON)
    def test_passes_then_fails_on_text_byte_and_corrupt_bundle(self):
        bundle = self.copy_bundle()
        out = os.path.join(self.tmp, "ok")
        r = self.milestone(out, bundle)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("gate PASS 1/1", r.stdout)
        m = jload(os.path.join(out, "ref", "vecadd_macos.cubin.json"))
        self.assertEqual(m["cubin_sha256"], sha(os.path.join(out, "ref", "vecadd_macos.cubin")))
        self.assertTrue(m["identity"]["verified"])
        # flip one .text byte of the reference: SHA no longer matches the bundle list -> FAIL
        ref = os.path.join(bundle, "cubins", "vecadd_macos.cubin")
        s = next(x for x in cubin_diff.parse(ref)["secs"] if x["name"] == ".text.vecadd_macos")
        b = bytearray(rd(ref))
        b[s["off"] + 32] ^= 0xFF
        wr(ref, bytes(b))
        r = self.milestone(os.path.join(self.tmp, "bad"), bundle)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("BUNDLE SHA MISMATCH", r.stdout)
        self.assertIn("gate FAIL", r.stdout)
        self.assertIn("VIOLATION: .text.vecadd_macos", r.stdout)


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------
def fake_elf(machine):
    """64-byte ELF header that `file` reports as an executable for `machine`."""
    return (b"\x7fELF\x02\x01\x01\x00" + b"\0" * 8 +
            struct.pack("<HHIQQQIHHHHHH", 2, machine, 1, 0x400000, 64, 0, 0, 64, 56, 0, 64, 0, 0)) + b"\0" * 64


class ResolverBase(Tmp):
    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        os.makedirs(self.home)
        self.marker = os.path.join(self.tmp, "EXECUTED")
        self.fake_dir = os.path.join(self.tmp, "pathbin")
        os.makedirs(self.fake_dir)
        fake = os.path.join(self.fake_dir, "ptxas")
        wr(fake, f"#!/bin/sh\ntouch '{self.marker}'\necho fake\n", "w")
        os.chmod(fake, 0o755)
        self.env = dict(PATH=self.fake_dir + os.pathsep + os.environ.get("PATH", ""), HOME=self.home,
                        PTXAS_NATIVE_ARCHIVE=os.path.expanduser(
                            "~/.config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/"
                            "libnvptxcompiler-linux-sbsa-13.2.51-archive/lib/libnvptxcompiler_static.a"),
                        PTXAS_NATIVE_X18_MANIFEST=os.path.expanduser(
                            "~/.cache/ptxas-native/13.2.51/x18_patches.tsv"))

    def resolve(self, *args, env=None):
        e = dict(self.env)
        e.update(env or {})
        return run(["bash", RESOLVE, *args], env=e)

    def assert_never_executed(self):
        self.assertFalse(os.path.exists(self.marker), "a PATH ptxas was executed")

    def pin(self, binary_sha, archive=None, x18=None):
        ident = jload_str(run([NATIVE_BIN, "--identity"], env=self.env).stdout) if NATIVE_OK else {}
        p = os.path.join(self.home, ".config", "cuda-oxide")
        os.makedirs(p, exist_ok=True)
        wr(os.path.join(p, "ptxas-native.pin"),
           f"binary_sha256={binary_sha}\narchive_sha256={archive or ident.get('archive_sha256', 'a' * 64)}\n"
           f"x18_manifest_sha256={x18 or ident.get('x18_manifest_sha256', 'b' * 64)}\n", "w")


def jload_str(s):
    return json.loads(s)


class SpdxHeaders(unittest.TestCase):
    def test_every_source_file_has_an_spdx_identifier(self):
        r = run(["bash", os.path.join(SCRIPTS, "check_spdx.sh")])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SPDX: PASS", r.stdout)


class ResolverCommon(ResolverBase):
    def test_usage_errors(self):
        self.assertEqual(self.resolve("--print-path", "extra").returncode, 2)
        self.assertEqual(self.resolve().returncode, 2)


@unittest.skipUnless(IS_MAC_ARM, "macOS arm64 contract (Linux contract is tested via simulation below)")
class ResolverDarwin(ResolverBase):
    def test_missing_native_fails_closed_without_fallback(self):
        nope = dict(PTXAS_DARWIN_ARM64_BIN=os.path.join(self.tmp, "nope"))
        for mode in ("--print-path", "--print-identity"):
            r = self.resolve(mode, env=nope)
            self.assertEqual((r.returncode, r.stdout), (3, ""), r.stderr)
            self.assertIn("ptxas-resolve:", r.stderr)
        self.assertEqual(self.resolve("--gpu-name=sm_120", "x.ptx", env=nope).returncode, 3)
        self.assert_never_executed()

    def test_status_never_executes_path_ptxas(self):
        r = self.resolve("--status")
        self.assertEqual(r.returncode, 0)
        self.assertIn("not selected, not executed", r.stdout)
        self.assert_never_executed()

    def test_no_pin_file_rejected(self):
        if not os.path.exists(NATIVE_BIN):
            self.skipTest(NATIVE_REASON)
        r = self.resolve("--print-path")
        self.assertEqual((r.returncode, r.stdout), (3, ""))
        self.assertIn("no pin file", r.stderr)

    def test_any_macho_not_matching_pin_rejected(self):
        # /bin/ls is a perfectly good Mach-O arm64 binary -- but not the pinned one
        self.pin("0" * 64)
        r = self.resolve("--print-path", env=dict(PTXAS_DARWIN_ARM64_BIN="/bin/ls"))
        self.assertEqual((r.returncode, r.stdout), (3, ""))
        self.assertIn("!= pinned", r.stderr)
        # even when the pin names it, a binary whose identity does not verify is refused
        self.pin(sha("/bin/ls"))
        r = self.resolve("--print-path", env=dict(PTXAS_DARWIN_ARM64_BIN="/bin/ls"))
        self.assertEqual((r.returncode, r.stdout), (3, ""), r.stderr)
        self.assert_never_executed()

    def test_non_macho_candidate_rejected(self):
        fake = os.path.join(self.fake_dir, "ptxas")
        self.pin(sha(fake))
        r = self.resolve("--print-path", env=dict(PTXAS_DARWIN_ARM64_BIN=fake))
        self.assertEqual(r.returncode, 3)
        self.assertIn("Mach-O", r.stderr)
        self.assert_never_executed()

    @require(NATIVE_OK, NATIVE_REASON)
    def test_pinned_native_selected_as_private_copy(self):
        self.pin(sha(NATIVE_BIN))
        env = dict(PTXAS_DARWIN_ARM64_BIN=NATIVE_BIN)
        r = self.resolve("--print-path", env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 1)
        path = lines[0]
        priv = os.path.realpath(os.path.join(self.home, ".cache", "ptxas-native", "bin"))
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(os.path.realpath(path).startswith(priv + os.sep), path)
        self.assertEqual(sha(path), sha(NATIVE_BIN))
        self.assertEqual(os.stat(os.path.dirname(path)).st_mode & 0o077, 0, "private dir must be 0700")
        r = self.resolve("--print-identity", env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        ident = json.loads(r.stdout)
        self.assertEqual(ident["schema"], "cuda-oxide.ptxas-identity.v1")
        self.assertTrue(ident["verified"] and ident["archive_pinned"] and ident["x18_manifest_pinned"])
        self.assertEqual(ident["binary_path"], path)
        self.assertEqual(ident["binary_sha256"], sha(path))
        # swapping the build-tree binary after pinning is refused (hash of the copy != pin)
        swapped = self.put("swapped", rd(NATIVE_BIN) + b"\0")
        os.chmod(swapped, 0o755)
        r = self.resolve("--print-path", env=dict(PTXAS_DARWIN_ARM64_BIN=swapped))
        self.assertEqual((r.returncode, r.stdout), (3, ""))
        self.assert_never_executed()

    @require(NATIVE_OK, NATIVE_REASON)
    def test_unpinned_archive_rejected(self):
        self.pin(sha(NATIVE_BIN))
        bogus = self.put("lib.a", b"!<arch>\n")
        r = self.resolve("--print-path", env=dict(PTXAS_DARWIN_ARM64_BIN=NATIVE_BIN, PTXAS_NATIVE_ARCHIVE=bogus))
        self.assertEqual((r.returncode, r.stdout), (3, ""))
        self.assertIn("pinned", r.stderr)

    @require(NATIVE_OK, NATIVE_REASON)
    def test_passthrough_compiles_validates_and_is_deterministic(self):
        self.pin(sha(NATIVE_BIN))
        env = dict(PTXAS_DARWIN_ARM64_BIN=NATIVE_BIN)
        out = os.path.join(self.tmp, "v.cubin")
        ptx = os.path.join(FX_PTX, "vecadd_macos.ptx")
        r = self.resolve("--gpu-name=sm_120", ptx, "-o", out, env=dict(env, CUDA_OXIDE_PTXAS_MANIFEST="1"))
        self.assertEqual(r.returncode, 0, r.stderr)
        m = jload(out + ".json")
        self.assertEqual(m["cubin_sha256"], sha(out))
        self.assertEqual((m["cubin_sm"], m["kernels"]), (120, ["vecadd_macos"]))
        out2 = os.path.join(self.tmp, "v2.cubin")
        self.assertEqual(self.resolve("--gpu-name=sm_120", ptx, "-o", out2, env=env).returncode, 0)
        self.assertEqual(rd(out), rd(out2))
        self.assertFalse(os.path.exists(out2 + ".json"))
        # requesting a different SM than the PTX/cubin -> refused, nothing published,
        # and the stale manifest of an earlier run is removed
        wr(out2 + ".json", "{}", "w")
        r = self.resolve("--gpu-name=sm_90", ptx, "-o", out2, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(out2) or os.path.exists(out2 + ".json"))
        self.assert_never_executed()


class ResolverLinuxSimulated(ResolverBase):
    """The Linux branch, simulated on any host: fake CUDA roots holding a 64-byte ELF
    header (never executable here, and the resolver never tries)."""

    def linux(self, *args, cuda_home=None, host="Linux-x86_64"):
        env = dict(PTXAS_RESOLVE_TESTING="1", PTXAS_RESOLVE_TEST_HOST=host)
        if cuda_home is not None:
            env["CUDA_HOME"] = cuda_home
        return self.resolve(*args, env=env)

    def root(self, machine=62, mode=0o755, dir_mode=0o755):
        r = os.path.join(self.tmp, f"cuda-{machine}-{mode:o}-{dir_mode:o}")
        os.makedirs(os.path.join(r, "bin"))
        p = os.path.join(r, "bin", "ptxas")
        wr(p, fake_elf(machine))
        os.chmod(p, mode)
        os.chmod(os.path.join(r, "bin"), dir_mode)
        return r, p

    def test_trusted_root_selected_without_execution(self):
        r_, p = self.root()
        r = self.linux("--print-path", cuda_home=r_)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(os.path.realpath(r.stdout.strip()), os.path.realpath(p))
        r = self.linux("--print-identity", cuda_home=r_)
        ident = json.loads(r.stdout)
        self.assertEqual((ident["schema"], ident["wrapper"], ident["verified"]),
                         ("cuda-oxide.ptxas-identity.v1", "system-ptxas", True))
        self.assertEqual(ident["binary_sha256"], sha(p))
        self.assertEqual(self.linux("--status", cuda_home=r_).returncode, 0)
        self.assert_never_executed()

    def test_path_is_never_searched(self):
        # Only a PATH ptxas exists (CUDA_HOME points at an empty root) -> exit 3
        r = self.linux("--print-path", cuda_home=os.path.join(self.tmp, "empty"))
        self.assertEqual((r.returncode, r.stdout), (3, ""))
        self.assertIn("PATH is not searched", r.stderr)
        self.linux("--status", cuda_home=os.path.join(self.tmp, "empty"))
        self.assert_never_executed()

    def test_wrong_arch_and_loose_permissions_rejected(self):
        r_, _ = self.root(machine=183)  # aarch64 ELF on an x86_64 host
        r = self.linux("--print-path", cuda_home=r_)
        self.assertEqual(r.returncode, 3)
        self.assertIn("not an ELF executable for x86_64", r.stderr)
        self.assertEqual(self.linux("--print-path", cuda_home=r_, host="Linux-aarch64").returncode, 0)
        r_, _ = self.root(mode=0o777)
        r = self.linux("--print-path", cuda_home=r_)
        self.assertEqual(r.returncode, 3)
        self.assertIn("writable", r.stderr)
        r_, _ = self.root(dir_mode=0o777)
        self.assertEqual(self.linux("--print-path", cuda_home=r_).returncode, 3)

    def test_test_hook_requires_testing_flag(self):
        r_, _ = self.root()
        r = self.resolve("--print-path", env=dict(PTXAS_RESOLVE_TEST_HOST="Linux-x86_64", CUDA_HOME=r_))
        if IS_MAC_ARM:
            self.assertNotEqual(r.stdout.strip(), os.path.join(r_, "bin", "ptxas"),
                                "host override must be ignored without PTXAS_RESOLVE_TESTING=1")


if __name__ == "__main__":
    print(f"reference bundle: {BUNDLE_STATE} ({BUNDLE}); native compiler verified: {NATIVE_OK}", file=sys.stderr)
    unittest.main(verbosity=2)
