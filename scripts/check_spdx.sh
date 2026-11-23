#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: SPDX hygiene check for the solution-105 sources (operator decision
# 2026-11-09: "should be spdx"). Every source file under the paths below must carry an
# `SPDX-License-Identifier:` line in its first 10 lines, with an allowed identifier, and every
# Cargo.toml `license` must be the expected SPDX expression. Per-file headers are authoritative.
#   portkit/ptxas-native/scripts/check_spdx.sh            # prints the per-file list; exit 1 on any gap
# Skipped (not source): target/, __pycache__/, Cargo.lock (cargo-generated), .DS_Store.
set -euo pipefail
REPO="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "$REPO"
PATHS=(portkit/ptxas-native portkit/binary-ko-loader tools/ptxas-darwin-arm64 tools/ptxas-resolve.sh
       crates/cuda-driverkit/examples/cubin_inspect.rs)
ALLOWED='^(Apache-2\.0|MIT)$'
bad=0
while IFS= read -r f; do
  id=$(head -10 "$f" | sed -n 's/.*SPDX-License-Identifier: *\([A-Za-z0-9.+-]*\).*/\1/p' | head -1)
  if [ -z "$id" ]; then echo "MISSING  $f"; bad=$((bad+1))
  elif ! printf '%s' "$id" | grep -Eq "$ALLOWED"; then echo "BAD($id) $f"; bad=$((bad+1))
  else printf '%-11s %s\n' "$id" "$f"; fi
done < <(find "${PATHS[@]}" \( -name target -o -name __pycache__ \) -prune -o -type f \
           ! -name Cargo.lock ! -name .DS_Store -print | sort)
# Cargo.toml license expressions must match the per-file headers.
lic() { sed -n 's/^license *= *"\(.*\)"/\1/p' "$1" | head -1; }
[ "$(lic portkit/ptxas-native/Cargo.toml)" = "Apache-2.0" ] || { echo "BAD license in portkit/ptxas-native/Cargo.toml"; bad=$((bad+1)); }
loader_ids=$(for f in $(find portkit/binary-ko-loader/src -name '*.rs'); do head -10 "$f" | sed -n 's/.*SPDX-License-Identifier: *\([A-Za-z0-9.+-]*\).*/\1/p' | head -1; done | sort -u | tr '\n' ' ')
want="MIT AND Apache-2.0"; [ "$loader_ids" = "Apache-2.0 " ] && want="Apache-2.0"; [ "$loader_ids" = "MIT " ] && want="MIT"
[ "$(lic portkit/binary-ko-loader/Cargo.toml)" = "$want" ] || { echo "BAD license in portkit/binary-ko-loader/Cargo.toml (files: $loader_ids; want \"$want\")"; bad=$((bad+1)); }
if [ "$bad" -gt 0 ]; then echo "SPDX: FAIL ($bad problem(s))"; exit 1; fi
echo "SPDX: PASS"
