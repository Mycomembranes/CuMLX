#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: solution 105 Route B milestone (NO GPU). A GATE:
#  (0) binary-ko-loader's own unit tests (the in-process ELF loader + glibc shim);
#  (1) compile the 12 PTX inputs that have a Linux ptxas 13.2.78 reference cubin, with
#      the reference's own options (as recorded in its .note.nv.tkinfo), and gate with
#      `cubin_diff.py diff --strict-options`: section-identical except the
#      .note.nv.tkinfo tool-version + build strings (layout shift from that note only);
#  (2) compile the 16 no-reference PTX inputs and validate each cubin strictly
#      (`cubin_diff.py check --ptx`: ELF/SM/entries/.nv.info/KPARAM vs PTX params);
#  (3) parse every produced cubin with cuda-driverkit's launch-path parser
#      (cargo example `cubin_inspect`).
# Every produced cubin gets a `<cubin>.json` manifest (ptxas-darwin-arm64 --manifest).
#
# Inputs:
#   * our PTX sources: portkit/ptxas-native/fixtures/ptx/ (tracked);
#   * the EXTERNAL reference bundle (NVIDIA-compiled; never committed): the 12 Linux
#     reference cubins + 2 NVVM-generated PTX, listed with SHA-256 in
#     portkit/ptxas-native/fixtures/reference-bundle.sha256, under
#     $PTXAS_NATIVE_REF_BUNDLE (default crates/cuda-driverkit/test_assets).
#     Every bundle file is SHA-checked before use. If the bundle is ABSENT, step (1)
#     is SKIPPED (said loudly) and the run can at best end in SKIP, never PASS.
#
# Exit: 0 = PASS (every step ran and passed); 77 = SKIP (no failures, but the reference
# bundle was absent so the equality gate did not run); 1 = any failure (compile failure,
# gate violation, bundle file with a wrong SHA-256, structural problem, parser error,
# loader test failure); 2 = the compiler is not built/verified.
# Host-only: never touches a GPU, a daemon, or any eGPU port.
#
# Test hooks (used by portkit/ptxas-native/tests/test_gate_and_resolver.py):
#   PTXAS_NATIVE_REF_BUNDLE=<dir>  reference bundle root
#   MILESTONE_FIXTURES=<dir>       our PTX sources (default portkit/ptxas-native/fixtures/ptx)
#   MILESTONE_ONLY=a,b             only these reference pairs
#   MILESTONE_SKIP_NOREF=1  MILESTONE_SKIP_INSPECT=1  MILESTONE_SKIP_LOADER_TESTS=1
set -euo pipefail

REPO="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
HERE="$REPO/portkit/ptxas-native"
BIN="${PTXAS_DARWIN_ARM64:-$HERE/target/release/ptxas-darwin-arm64}"
OUT="${1:-$HOME/.cache/ptxas-native/13.2.51/milestone}"
FX="${MILESTONE_FIXTURES:-$HERE/fixtures/ptx}"
BUNDLE="${PTXAS_NATIVE_REF_BUNDLE:-$REPO/crates/cuda-driverkit/test_assets}"
BUNDLE_SHA="$HERE/fixtures/reference-bundle.sha256"
DIFF=(python3 -I "$HERE/scripts/cubin_diff.py")
mkdir -p "$OUT/ref" "$OUT/noref" "$OUT/src"
rm -f "$OUT"/ref/*.cubin "$OUT"/ref/*.json "$OUT"/noref/*.cubin "$OUT"/noref/*.json

# libnvptxcompiler 13.2.51 rejects non-ASCII bytes even inside // comments
# ("Unexpected non-ASCII character"); several PTX files carry em-dashes /
# arrows / section signs in comments. Compile an ASCII-sanitized COPY (the
# source file is never modified); report which inputs needed it.
ascii_copy() {
  local src="$1" dst
  dst="$OUT/src/$(basename "$1")"
  if LC_ALL=C grep -q '[^[:print:][:space:]]' "$src"; then
    LC_ALL=C sed 's/[^[:print:][:space:]]/?/g' "$src" >"$dst"
    echo "    (sanitized non-ASCII comment bytes: $(basename "$src"))" >&2
  else
    cp "$src" "$dst"
  fi
  echo "$dst"
}
[ -x "$BIN" ] || { echo "missing $BIN (run portkit/ptxas-native/build.sh)" >&2; exit 2; }
"$BIN" --identity >"$OUT/identity.json" || { cat "$OUT/identity.json" >&2; echo "compiler not verified" >&2; exit 2; }

fail=0
loader="skipped"
if [ "${MILESTONE_SKIP_LOADER_TESTS:-0}" != 1 ]; then
  echo "== (0) binary-ko-loader unit tests =="
  if ( cd "$REPO" && cargo test -q --offline -p binary-ko-loader ) >"$OUT/loader_tests.txt" 2>&1; then
    loader="PASS"
  else
    loader="FAIL"; fail=$((fail+1))
  fi
  grep -E "test result" "$OUT/loader_tests.txt" | sed 's/^/  /' || tail -5 "$OUT/loader_tests.txt"
  echo "  loader tests: $loader"
fi

# Reference bundle: verify every listed file's SHA-256. Absent -> SKIP; wrong -> FAIL.
bundle="present"; missing=0; badsha=0
while read -r sha rel; do
  case "$sha" in ''|'#'*) continue ;; esac
  f="$BUNDLE/$rel"
  if [ ! -f "$f" ]; then missing=$((missing+1)); continue; fi
  if [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" != "$sha" ]; then
    echo "  BUNDLE SHA MISMATCH: $f (expected $sha)"; badsha=$((badsha+1))
  fi
done <"$BUNDLE_SHA"
if [ "$badsha" -gt 0 ]; then bundle="corrupt"; fail=$((fail+badsha)); fi
if [ "$missing" -gt 0 ]; then bundle="absent"; fi

# name|reference cubin (bundle)|ptx (fx: ours, bundle: NVVM-generated)|ptxas options (as recorded in the reference tkinfo)
PAIRS=(
  "vecadd_macos|cubins/vecadd_macos.cubin|fx:vecadd_macos.ptx|--gpu-name=sm_120"
  "vecadd_macos_v2|cubins/vecadd_macos_v2.cubin|fx:vecadd_macos_v2.ptx|-O3 --gpu-name=sm_120"
  "fused_rms_norm|cubins/fused_rms_norm.cubin|fx:fused_rms_norm.ptx|--gpu-name=sm_120"
  "fused_rms_norm_fixed|cubins/fused_rms_norm_fixed.cubin|fx:fused_rms_norm_fixed.ptx|--gpu-name=sm_120"
  "fused_swiglu|cubins/fused_swiglu.cubin|fx:fused_swiglu.ptx|--gpu-name=sm_120"
  "matmul_fp16_mma|cubins/matmul_fp16_mma.cubin|fx:matmul_fp16_mma.ptx|-O3 --gpu-name=sm_120"
  # NB: this reference was built from matmul_pipelined.ptx (same entry name, same
  # timestamp as the cubin); kernels/matmul_macos_pipelined.ptx is an older variant.
  "matmul_macos_pipelined|cubins/matmul_macos_pipelined.cubin|fx:matmul_pipelined.ptx|-O3 --gpu-name=sm_120"
  "matmul_macos|cubins/matmul_macos.cubin|fx:matmul_macos.ptx|--gpu-name=sm_120"
  "quantum_encode|cubins/quantum_encode.cubin|fx:quantum_encode.ptx|--gpu-name=sm_120"
  "rms_norm_v2|cubins/rms_norm_v2.cubin|bundle:kernels/rms_norm_v2.ptx|--gpu-name=sm_120 -m64"
  "softmax_v1|cubins/softmax_v1.cubin|bundle:kernels/softmax_v1.ptx|--gpu-name=sm_120 -m64"
  "vq_distance|cubins/vq_distance.cubin|fx:vq_distance.ptx|--gpu-name=sm_120"
)

compiled=0; gated=0; textsame=0; npairs=0
if [ "$bundle" = absent ]; then
  echo "== (1) reference pairs: SKIPPED — reference bundle absent ($missing of $(grep -vc '^#' "$BUNDLE_SHA") files missing under $BUNDLE)"
  echo "   install it as described in portkit/ptxas-native/README.md ('Reference bundle'); this run cannot PASS without it"
else
  echo "== (1) reference pairs (Linux ptxas 13.2.78 vs native libnvptxcompiler 13.2.51) =="
  for p in "${PAIRS[@]}"; do
    IFS='|' read -r name ref ptx opts <<<"$p"
    if [ -n "${MILESTONE_ONLY:-}" ] && [[ ",$MILESTONE_ONLY," != *",$name,"* ]]; then continue; fi
    npairs=$((npairs+1))
    case "$ptx" in fx:*) ptx="$FX/${ptx#fx:}" ;; bundle:*) ptx="$BUNDLE/${ptx#bundle:}" ;; esac
    src=$(ascii_copy "$ptx")
    # shellcheck disable=SC2086
    if "$BIN" --manifest $opts "$src" -o "$OUT/ref/$name.cubin" 2>"$OUT/ref/$name.log"; then
      compiled=$((compiled+1))
      if "${DIFF[@]}" diff --strict-options "$OUT/ref/$name.cubin" "$BUNDLE/$ref" >"$OUT/ref/$name.diff" 2>&1; then
        gated=$((gated+1)); verdict="PASS"
      else
        fail=$((fail+1)); verdict="FAIL"
      fi
      grep -q "all .text identical: True" "$OUT/ref/$name.diff" && textsame=$((textsame+1))
      echo "  $name: gate $verdict  ($(sed -n 3p "$OUT/ref/$name.diff"))"
      grep -E "VIOLATION|^cubin_diff:" "$OUT/ref/$name.diff" | sed 's/^/      /' || true
    else
      echo "  $name: COMPILE FAILED ($(head -c 300 "$OUT/ref/$name.log"))"; fail=$((fail+1))
    fi
  done
  [ "$npairs" -gt 0 ] || { echo "  no reference pairs selected (MILESTONE_ONLY=${MILESTONE_ONLY:-})"; fail=$((fail+1)); }
  echo "  compiled $compiled/$npairs, gate PASS $gated/$npairs (section-identical except .note.nv.tkinfo version/build), .text identical $textsame/$npairs"
fi

NOREF=(hybrid_ops_kernels cross_entropy_backward cross_entropy_backward_bf16 ecam_matmul_vjp
       entanglement_pairwise fused_cross_entropy fused_rotary_emb kernels rms_debug_sentinel
       rms_norm_backward rms_norm_backward_bf16 rotary_emb_backward rotary_emb_backward_bf16
       silu_backward silu_backward_bf16 topological_fingerprint)
nr_ok=0; nr_fail=0
if [ "${MILESTONE_SKIP_NOREF:-0}" != 1 ]; then
  echo "== (2) no-reference PTX (strict structural validation vs PTX .entry params) =="
  for n in "${NOREF[@]}"; do
    f="$FX/$n.ptx"
    [ -f "$f" ] || { echo "  $n: MISSING $f"; nr_fail=$((nr_fail+1)); continue; }
    src=$(ascii_copy "$f")
    if "$BIN" --manifest --gpu-name=sm_120 -O3 "$src" -o "$OUT/noref/$n.cubin" 2>"$OUT/noref/$n.log"; then
      if "${DIFF[@]}" check "$OUT/noref/$n.cubin" --ptx "$src" >"$OUT/noref/$n.check.json"; then
        k=$(python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]))[0];print(len(d['kernels']),'entries, params',sum(v['kparams'] for v in d['kernels'].values()))" "$OUT/noref/$n.check.json")
        echo "  $n: OK ($k)"; nr_ok=$((nr_ok+1))
      else
        echo "  $n: STRUCTURAL PROBLEMS: $(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))[0]['problems'])" "$OUT/noref/$n.check.json" 2>&1)"; nr_fail=$((nr_fail+1))
      fi
    else
      echo "  $n: COMPILE/VALIDATION FAILED: $(head -c 400 "$OUT/noref/$n.log" | tr '\n' ' ')"; nr_fail=$((nr_fail+1))
    fi
  done
  echo "  ok $nr_ok, failed $nr_fail"
fi

inspect="skipped"
if [ "${MILESTONE_SKIP_INSPECT:-0}" != 1 ]; then
  echo "== (3) cuda-driverkit launch-path parser (src/cubin.rs) =="
  shopt -s nullglob
  cubins=("$OUT"/ref/*.cubin "$OUT"/noref/*.cubin)
  shopt -u nullglob
  if [ "${#cubins[@]}" -eq 0 ]; then
    echo "  no cubins to parse"
  elif ( cd "$REPO" && cargo run -q --offline -p cuda-driverkit --example cubin_inspect -- "${cubins[@]}" ) >"$OUT/cubin_inspect.txt" 2>&1 \
     && ! grep -q "^ERR" "$OUT/cubin_inspect.txt"; then
    inspect="PASS"
  else
    inspect="FAIL"; fail=$((fail+1))
  fi
  [ -f "$OUT/cubin_inspect.txt" ] && { grep -E "^ERR|summary" "$OUT/cubin_inspect.txt" || tail -5 "$OUT/cubin_inspect.txt"; }
  echo "  launch-path parser: $inspect"
fi

# operator hand-off list: every produced cubin + sha256 (bit-check later on silicon)
( cd "$OUT" && shopt -s nullglob && files=(ref/*.cubin noref/*.cubin) && { [ "${#files[@]}" -eq 0 ] || shasum -a 256 "${files[@]}"; } ) >"$OUT/CUBINS_FOR_OPERATOR.sha256"
echo "operator list: $OUT/CUBINS_FOR_OPERATOR.sha256 ($(wc -l <"$OUT/CUBINS_FOR_OPERATOR.sha256" | tr -d ' ') cubins; manifests: <cubin>.json)"
total_fail=$((fail+nr_fail))
if [ "$total_fail" -gt 0 ]; then verdict=FAIL; rc=1
elif [ "$bundle" = absent ]; then verdict=SKIP; rc=77
else verdict=PASS; rc=0; fi
printf '{"schema":"cuda-oxide.ptxas-milestone.v2","verdict":"%s","reference_bundle":"%s","pairs":%d,"compiled":%d,"gate_pass":%d,"text_identical":%d,"noref_ok":%d,"noref_fail":%d,"launch_parser":"%s","loader_tests":"%s"}\n' \
  "$verdict" "$bundle" "$npairs" "$compiled" "$gated" "$textsame" "$nr_ok" "$nr_fail" "$inspect" "$loader" >"$OUT/milestone.json"
echo "MILESTONE: $verdict  (verdict JSON: $OUT/milestone.json)"
if [ "$verdict" = SKIP ]; then echo "  SKIP = reference bundle absent; the equality gate did NOT run (not a pass)"; fi
exit "$rc"
