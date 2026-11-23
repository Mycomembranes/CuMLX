SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

tileiras-native — NVIDIA tileiras (Tile IR compiler) on macOS arm64
==================================================================

STATUS (2026-11-09, macOS 26.6.2, measured in this tree)
  Runs natively, in an isolated helper process:
  * all 678 static constructors of NVIDIA's Linux-aarch64 tileiras 13.2.51;
  * `tileiras --version` / `--help` (NVIDIA's own text, exit 0), input errors;
  * NVIDIA's linux-sbsa libnvvm.so.4 (ET_DYN) loaded through the guest's
    dlopen: 65,996 relocations applied exactly, 741 constructors, nvvmVersion = 2.0;
  * Tile IR bytecode -> libNVVM -> PTX -> pinned native ptxas -> sm_120 cubin for
    two small kernels (add_f32, vecadd_f32) written with cuda-tile's pure-Python
    bytecode writer.
  * determinism and attribution (test vector_add_compile_is_deterministic_...):
    two compiles in fresh directories give byte-identical cubins; the PTX handed
    to ptxas equals tests/fixtures/vecadd_f32.expected.ptx.txt; assembling that
    PTX with the native ptxas alone gives the same .text and .nv.constant0 (the
    tileiras run only adds .nv.host sections from its --nv-host object).
  NOT established:
  * correctness of the cubins. They are validated STRUCTURALLY only (ELF header,
    .text.<kernel>, .nv.info.<kernel>, parameter layout). They were never run on
    a GPU and never compared with the output of tileiras on Linux;
  * anything beyond those two kernels (no FP16 GEMM, no real cuda.tile program:
    the cuda.tile front end needs a Linux C extension and does not run here);
  * the data-pointer rebase is still a heuristic (numbers below);
  * C++ exceptions cannot cross helper frames (they end in a named exit 72).
  Nothing here touches a GPU, a driver, a container, a VM or a remote machine.

Two programs
  tileiras-native          planner / auditor. Maps nothing executable, runs nothing.
      plan                 validate the pinned ELF, print a load plan
      rebase-audit         classify every data word the rebase rule slides (JSON)
  tileiras-native-helper   the ONLY process that executes NVIDIA code.
      --map-check          map + relocate + patch + protect, verify protections
                           with mach_vm_region, JSON receipt. No NVIDIA code.
      --unwind-gate-check island|native
                           prove the C++ unwind gate fires for a helper frame
                           (exit 72, named) and not for guest pcs. No NVIDIA code.
      --nvvm-check         load libNVVM as the guest's dlopen would, run its
                           constructors, call nvvmVersion, JSON receipt.
      [--ptxas-resolver tools/ptxas-resolve.sh] -- <tileiras args>
                           run the guest. 60 s alarm + CPU limit (--timeout),
                           --scratch DIR = working directory AND TMPDIR, explicit
                           environment (PATH HOME TERM TMPDIR LANG=C).
      exit 70 = unimplemented import / non-admitted dlopen or spawn (named);
      exit 71 = guest fault (pc / address / "unrebased link-time address");
      exit 72 = a C++ exception would have to unwind through a helper frame.

Compile example (after build.sh; paths relative to the repo root)
  S=$PWD/portkit/tileiras-native/target/scratch/demo; mkdir -p $S
  python3 -I portkit/tileiras-native/scripts/make_tileir_fixture.py \
      --cuda-tile-root portkit/tileiras-native/target/pip/unpacked \
      --kernel vecadd_f32 --out $S/vecadd_f32.tilebc
  portkit/tileiras-native/target/release/tileiras-native-helper --scratch $S \
      --ptxas-resolver $PWD/tools/ptxas-resolve.sh -- \
      --gpu-name=sm_120 vecadd_f32.tilebc -o vecadd_f32.cubin
  cargo run -p cuda-driverkit --example cubin_inspect -- $S/vecadd_f32.cubin   # file-only parser
  TILEIRAS_NATIVE_KEEP_PTX=<file> keeps a copy of the PTX libNVVM produced.

Build and verify (prereq -> download -> patch -> compile -> verify)
  FETCH=1 portkit/tileiras-native/build.sh     # first time: libNVVM + cuda-tile wheel
  portkit/tileiras-native/build.sh             # offline afterwards
Tests
  cargo test --offline --manifest-path portkit/tileiras-native/Cargo.toml
  TILEIRAS_NATIVE_TEST_ELF=$HOME/.config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/cuda_tileiras-linux-sbsa-13.2.51-archive/bin/tileiras \
    cargo test --offline --manifest-path portkit/tileiras-native/Cargo.toml -- --include-ignored
  python3 -m pytest -q -p no:cacheprovider scripts/tests/test_egpu_tileiras_port_audit.py
  (the ignored Rust tests need build.sh's outputs: both manifests, libNVVM, the
   unpacked wheel under target/pip/, a verified native ptxas. They execute NVIDIA
   code in the helper process only, each run under a 60 s wall clock, with a
   scratch directory under target/ as cwd and TMPDIR.)

Pinned inputs (never modified, never committed)
  tileiras 13.2.51 linux-sbsa, 90,527,008 bytes
    SHA256 297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a
  patch manifest ~/.cache/tileiras-native/13.2.51/patches.tsv (generated, deterministic)
    SHA256 9752c4f79fcc46640af1d2f293c2135c6c9d7d3cad2caa7c014f85b43782fb84
  libNVVM linux-sbsa 13.2.51 archive
    https://developer.download.nvidia.com/compute/cuda/redist/libnvvm/linux-sbsa/libnvvm-linux-sbsa-13.2.51-archive.tar.xz
    SHA256 366da4383992b8d13df63ffecd3a8315815955f50da0c0288ca5123212f67e02 (redistrib_13.2.0.json)
    nvvm/lib64/libnvvm.so.4.0.0 SHA256 f3cb2690f62c668924a373482a9bc748ace90b94bf13adce8e76a1f077682077
  libNVVM patch manifest ~/.cache/tileiras-native/13.2.51/libnvvm.patches.tsv
    SHA256 c3facbc0b7d78e729b9d8c259c2ef64beb5da7d3f9c22602f246c62268fd6ac0
  cuda_tile-1.8.0-cp312-cp312-manylinux2014_aarch64.whl (PyPI, Apache-2.0; test input only)
    SHA256 530232e8080c4d831e433d2e50bda0b8e4a2d5f803d9c1a878191852babd2752
  native ptxas: whatever tools/ptxas-resolve.sh --print-path verifies (solution 105)

Problem / Root Cause / Solution
  Problem      ET_EXEC is linked at 0x400000; it cannot be mapped there.
  Root cause   Darwin arm64 kills at exec any image whose __PAGEZERO is below
               4 GiB (measured: -pagezero_size 0x4000..0x80000000 -> SIGKILL,
               ad-hoc signed, with or without entitlements); with the default
               __PAGEZERO mach_vm_allocate(0x400000, FIXED) = KERN_INVALID_ADDRESS.
  Solution     Load at link address + 0x70_0000_0000. Code is PC-relative
               (16.3 M instructions: 3 literal loads, all in glibc's _start).
               Data pointers: aligned words of .init_array .fini_array
               .data.rel.ro .got .got.plt .data whose value lies in the image
               are candidates (134,907). 157 of them are CONTRADICTED as
               addresses and left unslid (manifest `norebase` rows); 134,750
               get the slide. The set is SHA-bound in the manifest.
               This is a HEURISTIC (ET_EXEC keeps no relocation records).

  Rebase evidence (`tileiras-native rebase-audit`; test
  rebase_rule_is_cross_checked_and_its_uncorroborated_part_is_counted)
    contradicted, left unslid                     157
        value in no section (0x400000 bit masks)        4
        value in unwind metadata                         8
        value in code, not 4-aligned                    28
        value in code, not a function start            117
      (bit masks 0x400000..0x4000000 and the two Itanium-demangler operator
       tables. The previous rule slid them: wrong, silent. 0 of them is an
       address that code forms with ADRP+ADD / ADR.)
    slid                                      134,750
        in a pointer-only section (.got ...)        22,650   ELF section typing
        value = function start (FDE / PLT)          97,130
        value = address code forms (ADRP+ADD, ADR)   1,597
        value = address of another slid word         1,909
        value = start of a C string in .rodata       7,588
        AMBIGUOUS: in-range, no corroboration        3,876   (.data.rel.ro 3,791, .data 85;
                                                             targets: .rodata 3,562, .data.rel.ro 239,
                                                             .bss 39, .data 36)
    not looked at by the rule: 14,658 aligned in-range words of .rodata, 36 of
    .gcc_except_table, 19 of .eh_frame, 13 of .dynamic, and 138 in-range values
    at 4-aligned-not-8-aligned offsets of .data.rel.ro / .data (those samples
    are u32 bit masks).
  What the numbers do and do not show: the generator (Python) and the audit
  (Rust) derive the contradicted set independently and must agree. The 3,876
  ambiguous words are probably pointers to plain tables, but nothing proves it;
  an integer among them would be changed silently (only its upper 32 bits: the
  slide is a multiple of 2^32). A true pointer among the 157 unslid words, or
  among the words the rule does not look at, faults in the unmapped low 4 GiB
  when dereferenced or called and the helper names it (exit 71); one that is
  only compared would not fault. Corroboration is evidence, not proof.

  Problem      x18 is a normal temporary in the guest (9,543 instructions; 7,873 in libNVVM).
  Root cause   Darwin reserves x18 and zeroes it on context switches.
  Solution     Each instruction moves to an island that keeps "virtual x18" in
               a per-thread TSD slot; two scratch registers are parked in the
               Darwin red zone only while the island runs. No liveness
               analysis; flags untouched; shapes: plain, adrp, cbz/cbnz,
               tbz/tbnz, blr (a tail jump: return addresses stay in guest
               code). llvm-mc verifies every rewrite.

  Problem      `mrs xN, tpidr_el0` thread-pointer reads (592; 390 in libNVVM).
  Root cause   Darwin does not preserve a user TPIDR_EL0; TPIDRRO_EL0 is the
               kernel-owned pthread TSD base.
  Solution     Island: mrs xN, tpidrro_el0; and xN, xN, #~7; ldr xN, [xN, #key*8]
               -> a Linux variant-I thread block allocated per thread: 16-byte
               TCB, the executable's 772-byte TLS (zero template), then 256
               bytes reserved for libNVVM's static TLS (40 used), then errno.
               The helper checks the tsd[key] layout at start-up.

  Problem      libNVVM is a second image (dlopen).
  Root cause   No Linux ld.so exists in the process.
  Solution     dso.rs maps the pinned ET_DYN at a fixed free address and applies
               its relocation records exactly: 65,712 RELATIVE, 37 GLOB_DAT,
               242 JUMP_SLOT, 1 ABS64, 4 TLSDESC (static: the descriptor's
               resolver returns the block's offset from the thread pointer);
               any other type, REL/RELR tables or text relocations are refused.
               Its x18 / TPIDR_EL0 sites come from a second SHA-bound manifest
               (0 svc). DT_INIT + DT_INIT_ARRAY run once under a lock (a
               recursive dlopen from a constructor gets the handle).
               dlopen admits only libnvvm.so / libnvvm.so.4 / libnvvm.so.4.0.0
               (any directory) and always loads the pinned file; any other name,
               dlopen(NULL), and dlsym / dlclose on any other handle are named
               traps (exit 70). 224 of its imports are bound, 29 trap by name
               (fork, execve, pipe, kill, select, poll, iconv, ...).

  Problem      ptxas is a child process.
  Root cause   tileiras finds `ptxas` in PATH and posix_spawn()s it.
  Solution     posix_spawn shim: only a program whose basename is `ptxas` is
               admitted, and what runs is the native ptxas printed by
               `--ptxas-resolver <tools/ptxas-resolve.sh> --print-path`, with
               the guest's arguments (`--opt-level N` -> `--opt-level=N`),
               environment and file actions; wait4 / waitpid / wait translate
               the status (exit code, terminating / stop signal). The resolver
               path is made absolute before the helper changes directory, and
               the selected file is checked before every spawn against a trust
               root the resolver does not supply: binary_sha256 of
               ~/.config/cuda-oxide/ptxas-native.pin (written by
               portkit/ptxas-native/build.sh). It must be
               <that sha256>/ptxas-darwin-arm64 and its bytes must hash to it. Resolver failure (exit 3), no resolver or a
               mismatch = named fatal error, exit 70, no cubin. Any other
               program = named trap.
               fork / execv / execve stay traps; posix_spawnp is not imported.
               The helper puts a placeholder `ptxas` first in the guest's PATH
               (so the lookup succeeds); it is never executed and exits 97 if
               anything ever runs it. It is created with O_EXCL in a real
               directory, never written through a symbolic link.

  Problem      glibc 2.17 imports (276 in tileiras, 256 in libNVVM).
  Root cause   Different variadic ABI, struct layouts, constants, errno values.
  Solution     tileiras: 246 bound after review, 3 weak null, 27 trap by name
               (exit 70). errno: the guest has its own Linux-numbered errno
               cell; every bound function is called through a thunk that
               translates Darwin errno into it. Locale is "C" only with glibc's
               __locale_struct layout. SIG_DFL/SIG_IGN are installed; a real
               guest handler is stored and read back but never installed.
               No socket API is imported by either image.

  Problem      C++ exceptions and helper frames.
  Root cause   Import thunks and native callers of guest callbacks (qsort,
               pthread_once, key destructors, libNVVM constructors run by the
               dlopen shim) have no ELF unwind info.
  Solution     Not unwinding: a named stop. The guest's libgcc unwinder asks
               dl_iterate_phdr for the FDE of each pc; when no object claims a
               pc that lies in a helper island or in native code, the shim
               prints "C++ exception cannot unwind through a non-guest frame:
               pc ... is ..." and exits 72. main, constructors, thread start
               routines and exit handlers are entered through one trampoline
               whose pc range is the legitimate end of stack (std::terminate,
               as on Linux). The query's first word is read only if mapped and
               readable; a non-unwinder caller of dl_iterate_phdr whose first
               data word happened to be a helper / native code address would
               be stopped too (none is known in the pinned images, not proven).
               Verified with synthetic queries
               (--unwind-gate-check), NOT with a real guest exception: none of
               the runs here throws.

  Problem      Kernel entry.
  Root cause   No ld.so, no Linux process stack.
  Solution     Linux initial stack (argc/argv/envp/auxv) is copied to the native
               stack and control enters the ELF _start; __libc_start_main is a
               shim that runs the binary's own __libc_csu_init exactly once,
               then main, then exit handlers. 0 `svc` instructions in either
               image: every kernel interaction goes through shim.rs.

Tile IR input without Linux (what was found)
  * PyPI has `cuda-tile` 1.8.0 (Apache-2.0): wheels for manylinux2014 x86_64 /
    aarch64 and Windows only, cp311..cp315; no macOS wheel, no usable sdist
    (it is a wheel-stub). `cutile` and `nvidia-cuda-tile` do not exist on PyPI.
  * The wheel is not pure (cuda/tile/_cext.cpython-312-aarch64-linux-gnu.so),
    so `import cuda.tile` cannot work on macOS. Its bytecode WRITER,
    cuda/tile/_bytecode/*.py (11 files), is pure Python with stdlib imports only.
  * scripts/make_tileir_fixture.py loads only that sub-package from the UNPACKED
    wheel (never installed; cuda/tile/__init__.py and the C extension never run)
    and emits two hand-built kernels by calling the op encoders.
  * The tileiras redist archive ships no sample bytecode. The format is NVIDIA's
    own Tile IR bytecode (version tag V_13_2), not MLIR bytecode a local tool writes.

Segment facts: p_align 0x10000 > 16 KiB host pages, so text / RELRO / data get
separate protections. PT_GNU_RELRO ends at 0x5942000 (mid 16 KiB page): the
page 0x5940000..0x5944000 stays writable (inward rounding). PT_TLS/.tbss shares
address 0x5806868 with .init_array: TLS is a separate per-thread block and the
image bytes there are never zeroed. No page is ever writable and executable.

Known limitations
  * Rebase heuristic: see the evidence table. 3,876 slid words are uncorroborated.
  * Exceptions: named stop (exit 72), not propagation. Code that relies on an
    exception passing through qsort / pthread_once / a constructor will not work.
  * printf-family shims read the caller's stack and bypass the errno thunk.
  * The fault reporter covers the main thread; it formats with the allocator.
  * Guest signal handlers are not installed; the signal mask is bookkeeping.
  * RELRO tail 0x5940000..0x5942000 stays writable (reported by --map-check).
  * dlclose never unmaps; libNVVM constructors run with the dlopen lock held
    (another thread's dlopen waits; a constructor waiting for that thread
    would deadlock until the alarm, as with ld.so's load lock).
  * A process running as the same user can still swap the native ptxas between
    the helper's re-hash and exec (the resolver documents the same residue).
  * The static TLS reserve for libNVVM is 256 bytes (it needs 40); a different
    libNVVM build is refused by its SHA anyway.
  * Unverified against Linux: byte-for-byte output equality for any input, and
    the silent exit 1 for an unknown option.

Primary ABI reference:
  https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
The native PTXAS port (portkit/ptxas-native, binary-ko-loader/src/inproc) is not
modified by this work. This is host-only compiler work; no eGPU is involved.
