#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# derive_symbols.sh - list the cu* Driver API symbols exported by the installed
# NVIDIA CUDA.framework (read-only) into symbols/cuda_10_1.txt.
#
# Nothing in the framework is modified or copied; only the symbol names are read.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_BIN="${CUDA_FRAMEWORK_BIN:-/Library/Frameworks/CUDA.framework/Versions/A/CUDA}"
OUT="${1:-$SCRIPT_DIR/../symbols/cuda_10_1.txt}"

if [ ! -f "$FRAMEWORK_BIN" ]; then
    echo "CUDA framework binary not found: $FRAMEWORK_BIN" >&2
    echo "Set CUDA_FRAMEWORK_BIN, or keep the checked-in symbols/cuda_10_1.txt." >&2
    exit 2
fi

/usr/bin/nm -gU "$FRAMEWORK_BIN" | awk '{print $3}' | sed 's/^_//' | grep '^cu' | LC_ALL=C sort -u > "$OUT"
echo "wrote $(wc -l < "$OUT" | tr -d ' ') symbols to $OUT"
