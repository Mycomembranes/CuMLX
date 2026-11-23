#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# build.sh - Build the native macOS ARM64 ptxas (portkit solution 105, Route B)
#
# Modified for macOS ARM64: NVIDIA ships no macOS ptxas. This script turns the
# Linux-aarch64 (linux-sbsa) libnvptxcompiler_static.a from the CUDA 13.2.51 redist
# into a native CLI by loading it IN-PROCESS (portkit/binary-ko-loader::inproc):
# same ISA, OS shims only. No Docker/Podman/Lima/Colima/Rosetta/VMs, no sudo.
#
#   prereq  -> verify tools + the SHA-256 of the downloaded archive (never modified)
#   extract -> unpack the 412 ELF members into a cache dir (read-only copy)
#   x18     -> generate the x18 rewrite manifest (CFG + liveness + IPA-RA aware,
#              llvm-mc verified encodings) — Darwin zeroes user x18
#   build   -> cargo build --release (ptxas-darwin-arm64)
#   verify  -> self-test (load + ctors + version + archs), --identity must report
#              verified, and vecadd PTX -> sm_120 cubin must pass the cubin_diff gate
#              (section-identical to the Linux-ptxas reference except .note.nv.tkinfo);
#              then the binary's SHA-256 is pinned in ~/.config/cuda-oxide/ptxas-native.pin
#              (outside the binary) for tools/ptxas-resolve.sh
#
# Prerequisites (user-space):
#   - Xcode Command Line Tools, Rust (stable), python3
#   - Homebrew llvm@21 (llvm-objdump / llvm-readelf / llvm-mc / llvm-ar)
#   - the redist already fetched: tools/fetch-cuda-redist.sh (linux-sbsa 13.2.51)
#
# Usage:
#   portkit/ptxas-native/build.sh            # full pipeline
#   SKIP_X18=1 portkit/ptxas-native/build.sh # reuse an existing manifest
# Environment:
#   PTXAS_NATIVE_ARCHIVE   (default ~/.config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/
#                           libnvptxcompiler-linux-sbsa-13.2.51-archive/lib/libnvptxcompiler_static.a)
#   PTXAS_NATIVE_CACHE     (default ~/.cache/ptxas-native/13.2.51)  members + x18 manifest
#   LLVM_BIN               (default /opt/homebrew/opt/llvm@21/bin)
set -euo pipefail

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/../.." && pwd)"
ARCHIVE="${PTXAS_NATIVE_ARCHIVE:-$HOME/.config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/libnvptxcompiler-linux-sbsa-13.2.51-archive/lib/libnvptxcompiler_static.a}"
# Both pins are mirrored in src/identity.rs (ARCHIVE_SHA256 / X18_MANIFEST_SHA256); the
# generator is deterministic, so the manifest hash is pinned as well.
ARCHIVE_SHA256="d703609da635db3976a3bc228f34e62f66acffcdc6d4bc5df09ba3a383e716cd"
X18_MANIFEST_SHA256="5b0104356d6c62e03ae64b64ae75a9f75265b151f2da9da45df097085fd060e3"
CACHE="${PTXAS_NATIVE_CACHE:-$HOME/.cache/ptxas-native/13.2.51}"
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm@21/bin}"
BIN="$HERE/target/release/ptxas-darwin-arm64"

log() { printf '\033[1;34m[ptxas-native]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[ptxas-native] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. prerequisites -------------------------------------------------------
log "1/5 prerequisites"
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || die "macOS arm64 only (on Linux use the real ptxas)"
for t in llvm-objdump llvm-readelf llvm-mc llvm-ar; do
  [ -x "$LLVM_BIN/$t" ] || die "missing $LLVM_BIN/$t (brew install llvm@21)"
done
command -v cargo >/dev/null || die "missing cargo (rustup)"
command -v python3 >/dev/null || die "missing python3"
[ -f "$ARCHIVE" ] || die "missing $ARCHIVE (tools/fetch-cuda-redist.sh, linux-sbsa 13.2.51)"
have=$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)
[ "$have" = "$ARCHIVE_SHA256" ] || die "archive SHA-256 $have != pinned $ARCHIVE_SHA256 (the x18 manifest is archive-specific)"
log "   archive OK ($ARCHIVE_SHA256)"

# ---- 2. extract members (read-only copy) -----------------------------------
log "2/5 extract archive members -> $CACHE/members"
mkdir -p "$CACHE/members"
if [ ! -f "$CACHE/members/.done" ]; then
  ( cd "$CACHE/members" && "$LLVM_BIN/llvm-ar" x "$ARCHIVE" )
  touch "$CACHE/members/.done"
fi
log "   $(ls "$CACHE/members" | grep -c '\.o$') members"

# ---- 3. x18 manifest ---------------------------------------------------------
if [ "${SKIP_X18:-0}" != 1 ] || [ ! -f "$CACHE/x18_patches.tsv" ]; then
  log "3/5 x18 rewrite manifest (CFG + liveness + IPA-RA; ~90 s)"
  python3 -I "$HERE/scripts/gen_x18_patches.py" --members "$CACHE/members" \
      --out "$CACHE/x18_patches.tsv" --llvm-bin "$LLVM_BIN" | tee "$CACHE/x18_report.txt" | grep -v "UNPORTED\|  trap " || true
  [ "${PIPESTATUS[0]}" = 0 ] || die "x18 manifest generation failed"
else
  log "3/5 x18 manifest: reusing $CACHE/x18_patches.tsv"
fi
mhave=$(shasum -a 256 "$CACHE/x18_patches.tsv" | cut -d' ' -f1)
[ "$mhave" = "$X18_MANIFEST_SHA256" ] || die "x18 manifest SHA-256 $mhave != pinned $X18_MANIFEST_SHA256 (generator or archive changed; update both pins deliberately)"
log "   x18 manifest OK ($X18_MANIFEST_SHA256)"

# ---- 4. build ----------------------------------------------------------------
log "4/5 cargo build --release"
( cd "$HERE" && cargo build --release --quiet )
[ -x "$BIN" ] || die "build produced no $BIN"

# ---- 5. verify ---------------------------------------------------------------
log "5/5 verify"
PTXAS_NATIVE_X18_MANIFEST="$CACHE/x18_patches.tsv" PTXAS_NATIVE_ARCHIVE="$ARCHIVE" "$BIN" --self-test
tmp="$(mktemp -d)"
FX="$HERE/fixtures/ptx"
BUNDLE="${PTXAS_NATIVE_REF_BUNDLE:-$REPO/crates/cuda-driverkit/test_assets}"
REF="$BUNDLE/cubins/vecadd_macos.cubin"
PTXAS_NATIVE_X18_MANIFEST="$CACHE/x18_patches.tsv" PTXAS_NATIVE_ARCHIVE="$ARCHIVE" \
  "$BIN" --manifest --gpu-name=sm_120 "$FX/vecadd_macos.ptx" -o "$tmp/vecadd_macos.cubin"
PTXAS_NATIVE_X18_MANIFEST="$CACHE/x18_patches.tsv" PTXAS_NATIVE_ARCHIVE="$ARCHIVE" "$BIN" --identity >"$tmp/identity.json" \
  || { cat "$tmp/identity.json" >&2; die "--identity does not report a verified compiler"; }
python3 -I "$HERE/scripts/cubin_diff.py" check "$tmp/vecadd_macos.cubin" --ptx "$FX/vecadd_macos.ptx" >/dev/null \
  || die "vecadd cubin failed structural validation"
want=$(grep ' cubins/vecadd_macos.cubin$' "$HERE/fixtures/reference-bundle.sha256" | cut -d' ' -f1)
if [ -f "$REF" ]; then
  [ "$(shasum -a 256 "$REF" | cut -d' ' -f1)" = "$want" ] || die "reference $REF does not match fixtures/reference-bundle.sha256"
  python3 -I "$HERE/scripts/cubin_diff.py" diff --strict-options "$tmp/vecadd_macos.cubin" "$REF" \
    || die "vecadd cubin differs from the Linux ptxas reference outside the tolerated .note.nv.tkinfo fields"
  log "PASS: native PTX -> sm_120 cubin, section-identical to Linux ptxas 13.2.78 except .note.nv.tkinfo"
else
  log "WARNING: reference bundle absent ($REF): equality gate SKIPPED, structural validation only."
  log "         See README 'Reference bundle' to install it; scripts/milestone.sh reports SKIP (exit 77) until then."
fi
# Pin the verified binary OUTSIDE the binary: tools/ptxas-resolve.sh copies the binary
# into a private cache, hashes the copy and requires this value.
PIN="$HOME/.config/cuda-oxide/ptxas-native.pin"
mkdir -p "$(dirname "$PIN")"
{
  echo "# written by portkit/ptxas-native/build.sh after its verify step; read by tools/ptxas-resolve.sh"
  echo "binary_sha256=$(shasum -a 256 "$BIN" | cut -d' ' -f1)"
  echo "archive_sha256=$ARCHIVE_SHA256"
  echo "x18_manifest_sha256=$X18_MANIFEST_SHA256"
  echo "source_path=$BIN"
  echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$PIN.tmp.$$" && chmod 600 "$PIN.tmp.$$" && mv -f "$PIN.tmp.$$" "$PIN"
log "pinned binary: $PIN"
rm -rf "$tmp"
log "binary: $BIN   (tools/ptxas-darwin-arm64 wraps it; tools/ptxas-resolve.sh picks it up)"
log "full milestone: portkit/ptxas-native/scripts/milestone.sh"
