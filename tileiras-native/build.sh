#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# build.sh - Build the native macOS ARM64 tileiras worker (portkit solution 106)
#
# Modified for macOS ARM64: NVIDIA ships no macOS tileiras. This script prepares the
# Linux-aarch64 (linux-sbsa) `tileiras` 13.2.51 ET_EXEC for an isolated in-process
# worker: same ISA, OS shims only. No Docker/Podman/Lima/Colima/Rosetta/VMs, no sudo.
#
#   prereq   -> verify tools + the SHA-256 of the downloaded binary (never modified)
#   download -> tileiras comes from tools/fetch-cuda-redist.sh. With FETCH=1 the
#               linux-sbsa libNVVM archive (NVIDIA redist) and the cuda-tile wheel
#               (PyPI, only its pure-Python bytecode writer is used, to make a
#               test input) are fetched; both are SHA-256 verified. Without
#               FETCH=1 nothing touches the network and the compile check runs
#               only if those inputs are already present
#   patch    -> generate both patch manifests (tileiras: x18 / TPIDR_EL0 / svc /
#               absolute literals / data-pointer rebase rule + norebase rows;
#               libNVVM: x18 / TPIDR_EL0 / svc) and compare with the pins
#   compile  -> cargo build --release (tileiras-native, tileiras-native-helper)
#   verify   -> --map-check and --unwind-gate-check (no NVIDIA code), the real
#               `--version` banner, --nvvm-check (libNVVM loaded, nvvmVersion),
#               and - if the wheel and a verified native ptxas are present -
#               vector-add Tile IR -> sm_120 cubin, all in the helper process
#               under a wall-clock limit with target/scratch as cwd and TMPDIR
#
# Environment:
#   TILEIRAS_NATIVE_ELF    (default ~/.config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/
#                           cuda_tileiras-linux-sbsa-13.2.51-archive/bin/tileiras)
#   TILEIRAS_NATIVE_CACHE  (default ~/.cache/tileiras-native/13.2.51)
#   LLVM_BIN               (default /opt/homebrew/opt/llvm@21/bin)
#   FETCH=1                fetch + verify libnvvm-linux-sbsa-13.2.51 (40 MB) and
#                          cuda_tile-1.8.0 (manylinux2014_aarch64 wheel, unpacked
#                          under target/pip/, never installed)
set -euo pipefail

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REDIST="$HOME/.config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51"
ELF="${TILEIRAS_NATIVE_ELF:-$REDIST/extract/cuda_tileiras-linux-sbsa-13.2.51-archive/bin/tileiras}"
# The pins are mirrored in src/bin/tileiras-native-helper/{main,shim}.rs,
# scripts/make_tileir_fixture.py and README.txt.
ELF_SHA256="297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a"
MANIFEST_SHA256="9752c4f79fcc46640af1d2f293c2135c6c9d7d3cad2caa7c014f85b43782fb84"
LIBNVVM_URL="https://developer.download.nvidia.com/compute/cuda/redist/libnvvm/linux-sbsa/libnvvm-linux-sbsa-13.2.51-archive.tar.xz"
LIBNVVM_SHA256="366da4383992b8d13df63ffecd3a8315815955f50da0c0288ca5123212f67e02"  # redistrib_13.2.0.json
LIBNVVM_SO="$REDIST/extract/libnvvm-linux-sbsa-13.2.51-archive/nvvm/lib64/libnvvm.so.4.0.0"
LIBNVVM_SO_SHA256="f3cb2690f62c668924a373482a9bc748ace90b94bf13adce8e76a1f077682077"
LIBNVVM_MANIFEST_SHA256="c3facbc0b7d78e729b9d8c259c2ef64beb5da7d3f9c22602f246c62268fd6ac0"
WHEEL="cuda_tile-1.8.0-cp312-cp312-manylinux2014_aarch64.whl"
WHEEL_SHA256="530232e8080c4d831e433d2e50bda0b8e4a2d5f803d9c1a878191852babd2752"
PIP_DIR="$HERE/target/pip"
RESOLVER="$HERE/../../tools/ptxas-resolve.sh"
CACHE="${TILEIRAS_NATIVE_CACHE:-$HOME/.cache/tileiras-native/13.2.51}"
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm@21/bin}"
BIN="$HERE/target/release/tileiras-native-helper"

log() { printf '\033[1;34m[tileiras-native]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[tileiras-native] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. prerequisites -------------------------------------------------------
log "1/5 prerequisites"
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || die "macOS arm64 only (on Linux run the real tileiras)"
for t in llvm-objdump llvm-mc; do
  [ -x "$LLVM_BIN/$t" ] || die "missing $LLVM_BIN/$t (brew install llvm@21)"
done
command -v cargo >/dev/null || die "missing cargo (rustup)"
command -v python3 >/dev/null || die "missing python3"
[ -f "$ELF" ] || die "missing $ELF (tools/fetch-cuda-redist.sh, linux-sbsa 13.2.51)"
have=$(shasum -a 256 "$ELF" | cut -d' ' -f1)
[ "$have" = "$ELF_SHA256" ] || die "tileiras SHA-256 $have != pinned $ELF_SHA256 (the patch manifest is binary-specific)"
log "   tileiras OK ($ELF_SHA256)"

# ---- 2. download (optional) ---------------------------------------------------
if [ "${FETCH:-${FETCH_LIBNVVM:-0}}" = 1 ]; then
  log "2/5 libNVVM linux-sbsa 13.2.51 (NVIDIA redist) + cuda-tile 1.8.0 wheel (PyPI), SHA-256 pinned"
  ar="$REDIST/libnvvm-linux-sbsa-13.2.51-archive.tar.xz"
  if [ ! -f "$ar" ]; then
    curl -fsS -m 600 -o "$ar.part" "$LIBNVVM_URL" || die "libNVVM download failed (blocked network?)"
    mv "$ar.part" "$ar"
  fi
  [ "$(shasum -a 256 "$ar" | cut -d' ' -f1)" = "$LIBNVVM_SHA256" ] || die "libNVVM archive SHA-256 mismatch"
  [ -d "$REDIST/extract/libnvvm-linux-sbsa-13.2.51-archive" ] || tar -xJf "$ar" -C "$REDIST/extract"
  # The wheel is a Linux build (cp312, C extension): it is downloaded for a
  # foreign platform, verified, UNPACKED and never installed or imported as a
  # package. Only cuda/tile/_bytecode/*.py (pure Python) is ever loaded.
  wdir="$PIP_DIR/cuda-tile__manylinux2014_aarch64"
  if [ ! -f "$wdir/$WHEEL" ]; then
    mkdir -p "$wdir"
    python3 -I -m pip download --disable-pip-version-check -q --no-deps --only-binary :all: \
      --platform manylinux2014_aarch64 --python-version 312 --implementation cp -d "$wdir" cuda-tile==1.8.0 \
      || die "cuda-tile wheel download failed (blocked network?)"
  fi
  [ "$(shasum -a 256 "$wdir/$WHEEL" | cut -d' ' -f1)" = "$WHEEL_SHA256" ] || die "cuda-tile wheel SHA-256 mismatch"
  if [ ! -f "$PIP_DIR/unpacked/cuda/tile/_bytecode/writer.py" ]; then
    rm -rf "$PIP_DIR/unpacked" && mkdir -p "$PIP_DIR/unpacked"
    python3 -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$wdir/$WHEEL" "$PIP_DIR/unpacked"
  fi
  log "   libNVVM archive + cuda-tile wheel OK"
else
  log "2/5 download: skipped (FETCH=1 fetches libNVVM and the cuda-tile wheel)"
fi
[ -f "$LIBNVVM_SO" ] || die "missing $LIBNVVM_SO (run with FETCH=1)"
[ "$(shasum -a 256 "$LIBNVVM_SO" | cut -d' ' -f1)" = "$LIBNVVM_SO_SHA256" ] || die "libnvvm.so.4.0.0 SHA-256 mismatch"

# ---- 3. patch manifests ------------------------------------------------------
log "3/5 patch manifests (llvm-objdump + llvm-mc verified; ~40 s)"
mkdir -p "$CACHE"
python3 -I "$HERE/scripts/gen_patch_manifest.py" --elf "$ELF" --out "$CACHE/patches.tsv" --llvm-bin "$LLVM_BIN" \
  | grep -v '^rebase ' || die "patch manifest generation failed"
mhave=$(shasum -a 256 "$CACHE/patches.tsv" | cut -d' ' -f1)
[ "$mhave" = "$MANIFEST_SHA256" ] || die "manifest SHA-256 $mhave != pinned $MANIFEST_SHA256 (generator or llvm changed; update the pins deliberately)"
log "   tileiras manifest OK ($MANIFEST_SHA256)"
python3 -I "$HERE/scripts/gen_patch_manifest.py" --elf "$LIBNVVM_SO" --out "$CACHE/libnvvm.patches.tsv" --llvm-bin "$LLVM_BIN" \
  || die "libNVVM patch manifest generation failed"
mhave=$(shasum -a 256 "$CACHE/libnvvm.patches.tsv" | cut -d' ' -f1)
[ "$mhave" = "$LIBNVVM_MANIFEST_SHA256" ] || die "libNVVM manifest SHA-256 $mhave != pinned $LIBNVVM_MANIFEST_SHA256"
log "   libNVVM manifest OK ($LIBNVVM_MANIFEST_SHA256)"

# ---- 4. compile --------------------------------------------------------------
log "4/5 cargo build --release"
( cd "$HERE" && cargo build --release --offline --quiet )
[ -x "$BIN" ] || die "build produced no $BIN"
# The helper flips anonymous RW pages to R-X with mprotect; that needs a binary
# WITHOUT the hardened runtime (the default ad-hoc linker signature qualifies).
if codesign -dv "$BIN" 2>&1 | grep -q 'flags=.*runtime'; then die "helper is signed with the hardened runtime; W^X flips would fail"; fi

# ---- 5. verify ---------------------------------------------------------------
log "5/5 verify"
scratch="$HERE/target/scratch"; mkdir -p "$scratch"
"$BIN" --map-check --elf "$ELF" --manifest "$CACHE/patches.tsv" >"$scratch/map-check.json" || die "--map-check failed"
grep -q '"nvidia_code_executed":false' "$scratch/map-check.json" || die "map-check receipt malformed"
banner=$("$BIN" --elf "$ELF" --manifest "$CACHE/patches.tsv" --timeout 60 --scratch "$scratch" -- --version) \
  || die "guest --version failed"
printf '%s\n' "$banner" | grep -q 'Cuda compilation tools, release 13.2, V13.2.51' || die "unexpected version banner"
log "PASS: map-check + NVIDIA version banner:"
printf '%s\n' "$banner" | sed 's/^/        /'
for kind in island native; do
  set +e; "$BIN" --elf "$ELF" --manifest "$CACHE/patches.tsv" --unwind-gate-check "$kind" >/dev/null 2>"$scratch/gate-$kind.txt"; rc=$?; set -e
  [ "$rc" = 72 ] && grep -q 'C++ exception cannot unwind through a non-guest frame' "$scratch/gate-$kind.txt" \
    || die "--unwind-gate-check $kind: expected the named diagnostic and exit 72, got $rc"
done
log "PASS: unwind gate (helper frames end with the named diagnostic, exit 72)"
"$BIN" --elf "$ELF" --manifest "$CACHE/patches.tsv" --libnvvm "$LIBNVVM_SO" --libnvvm-manifest "$CACHE/libnvvm.patches.tsv" --timeout 60 --nvvm-check >"$scratch/nvvm-check.json" || die "--nvvm-check failed"
grep -q '"nvvmVersion_result":0,"nvvm_major":2,"nvvm_minor":0' "$scratch/nvvm-check.json" || die "unexpected nvvmVersion answer"
log "PASS: libNVVM loaded in the helper, nvvmVersion() = 2.0"

# Tile IR -> cubin: needs the unpacked cuda-tile wheel (test input) and a
# verified native ptxas. Each missing piece is named; nothing is faked.
if [ ! -f "$PIP_DIR/unpacked/cuda/tile/_bytecode/writer.py" ]; then
  log "SKIPPED: Tile IR -> cubin (no cuda-tile wheel under target/pip; run with FETCH=1)"
elif ! ptxas=$("$RESOLVER" --print-path 2>"$scratch/ptxas-resolve.txt"); then
  log "SKIPPED: Tile IR -> cubin (tools/ptxas-resolve.sh selected no verified native ptxas: $(tr '\n' ' ' <"$scratch/ptxas-resolve.txt"))"
else
  work="$scratch/compile"; rm -rf "$work"; mkdir -p "$work"
  python3 -I "$HERE/scripts/make_tileir_fixture.py" --cuda-tile-root "$PIP_DIR/unpacked" --kernel vecadd_f32 \
    --out "$work/vecadd_f32.tilebc" >/dev/null || die "Tile IR fixture generation failed"
  "$BIN" --elf "$ELF" --manifest "$CACHE/patches.tsv" --libnvvm "$LIBNVVM_SO" --libnvvm-manifest "$CACHE/libnvvm.patches.tsv" \
    --timeout 120 --scratch "$work" --ptxas-resolver "$RESOLVER" -- \
    --gpu-name=sm_120 vecadd_f32.tilebc -o vecadd_f32.cubin || die "Tile IR -> cubin failed"
  [ "$(head -c 4 "$work/vecadd_f32.cubin" | od -An -tx1 | tr -d ' \n')" = "7f454c46" ] || die "output is not an ELF"
  "$LLVM_BIN/llvm-readelf" -S -W "$work/vecadd_f32.cubin" >"$work/sections.txt" 2>/dev/null || die "llvm-readelf rejected the cubin"
  for sec in .text.vecadd_f32 .nv.info.vecadd_f32 .nv.info; do
    grep -q " $sec " "$work/sections.txt" || die "cubin has no $sec section"
  done
  log "PASS: vector-add Tile IR -> sm_120 cubin ($(wc -c <"$work/vecadd_f32.cubin" | tr -d ' ') bytes, structural check only; ptxas = $ptxas)"
fi
log "NOT verified: GPU execution of the cubin, equality with a Linux tileiras run"
log "binary: $BIN"
