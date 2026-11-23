#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# selftest.sh - host-only functional test of the adapter. Starts its OWN
# cuda_adapter_server with the loopback backend (host memory, no GPU) on a
# private socket, runs the clients through the stand-in library, stops it.
# Never connects to a served stack and never touches a card.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ADAPTER_DIR/../.." && pwd)"
BUILD="$ADAPTER_DIR/build"
HAVE_ROSETTA="${HAVE_ROSETTA:-1}"
HAVE_FORTRAN="${HAVE_FORTRAN:-1}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/cuda-adapter-selftest.XXXXXX")"
SOCK="$WORK/a.sock"
SERVER_PID=""
PASSED=0
FAILED=0

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

run() { # run <label> <expected exit> <command...>
    local label="$1" want="$2" rc=0
    shift 2
    "$@" > "$WORK/out.txt" 2>&1 || rc=$?
    if [ "$rc" = "$want" ]; then
        echo "[PASS] $label"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $label (exit $rc, expected $want)"
        sed 's/^/       /' "$WORK/out.txt" | tail -8
        FAILED=$((FAILED + 1))
    fi
}

# The stand-in must win over any other libcuda already on the caller's path.
export DYLD_LIBRARY_PATH="$BUILD"
export CUDA_ADAPTER_SOCKET="$SOCK"

CUBINS="$REPO_ROOT/crates/cuda-driverkit/test_assets/cubins"
KERNELS="$REPO_ROOT/crates/cuda-driverkit/test_assets/kernels"
VECADD_CUBIN="$CUBINS/vecadd_macos.cubin"
if [ ! -f "$VECADD_CUBIN" ]; then
    echo "[ERROR] $VECADD_CUBIN is missing (cubins are not tracked; build or copy the test assets)" >&2
    exit 1
fi

run "no server: cuInit reports CUDA_ERROR_NO_DEVICE" 3 "$BUILD/adapter_probe.arm64"

"$BUILD/cuda_adapter_server" --backend loopback --socket "$SOCK" > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
[ -S "$SOCK" ] || { echo "[ERROR] server did not start"; cat "$WORK/server.log"; exit 1; }

run "arm64 probe" 0 "$BUILD/adapter_probe.arm64"
run "arm64 vecadd (cubin)" 0 "$BUILD/vecadd_client.arm64" "$VECADD_CUBIN"
if [ "$HAVE_ROSETTA" = 1 ]; then
    run "x86_64 probe under Rosetta" 0 "$BUILD/adapter_probe.x86_64"
    run "x86_64 vecadd under Rosetta (cubin)" 0 "$BUILD/vecadd_client.x86_64" "$VECADD_CUBIN"
fi
if [ "$HAVE_FORTRAN" = 1 ] && [ -x "$BUILD/fortran_vecadd" ]; then
    run "Fortran vecadd (arm64)" 0 "$BUILD/fortran_vecadd" "$VECADD_CUBIN"
fi

# Shared host memory: the client's 80 MiB of cuMemAllocHost traffic must skip
# the socket, and must use it when shared segments are switched off.
check_log() { # check_log <label> <awk condition on s (socket bytes) and h (shared bytes)>
    local line socket shared
    sleep 0.3 # the summary is printed when the client disconnects
    line="$(grep "client done" "$WORK/server.log" | tail -1)"
    socket="$(echo "$line" | sed -E 's/.* ([0-9]+) copy bytes over the socket.*/\1/')"
    shared="$(echo "$line" | sed -E 's/.* ([0-9]+) through shared segments.*/\1/')"
    if awk -v s="$socket" -v h="$shared" "BEGIN { exit !($2) }"; then
        echo "[PASS] $1 (socket $socket bytes, shared $shared bytes)"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $1 (socket '$socket' bytes, shared '$shared' bytes)"
        FAILED=$((FAILED + 1))
    fi
}
run "arm64 vecadd, shared host segments" 0 "$BUILD/vecadd_client.arm64" "$VECADD_CUBIN"
check_log "host-memory copies skipped the socket" "h >= 80 * 1024 * 1024 && s < 1024 * 1024"
run "arm64 vecadd, shared segments off" 0 env CUDA_ADAPTER_NO_SHARED_HOST=1 "$BUILD/vecadd_client.arm64" "$VECADD_CUBIN"
check_log "without shared segments the bytes use the socket" "h == 0 && s >= 80 * 1024 * 1024"
if [ "$HAVE_ROSETTA" = 1 ]; then
    run "x86_64 vecadd under Rosetta, shared host segments" 0 "$BUILD/vecadd_client.x86_64" "$VECADD_CUBIN"
    check_log "Rosetta client mapped the segment and skipped the socket" "h >= 80 * 1024 * 1024 && s < 1024 * 1024"
fi

# PTX path: needs the repo's verified native ptxas. Reported as skipped, not passed, without it.
if "$REPO_ROOT/tools/ptxas-resolve.sh" --print-path >/dev/null 2>&1 && [ -f "$KERNELS/vecadd_macos.ptx" ]; then
    run "PTX compiled by the adapter (arm64 client)" 0 "$BUILD/vecadd_client.arm64" "$KERNELS/vecadd_macos.ptx" vecadd_macos_f32
    printf '.version 6.3\n.target sm_50\n.address_size 64\nnot ptx\n' > "$WORK/bad.ptx"
    run "broken PTX is refused" 1 "$BUILD/vecadd_client.arm64" "$WORK/bad.ptx"
else
    echo "[SKIP] PTX path (no verified ptxas on this host: portkit/ptxas-native/build.sh)"
fi

# cuBLASLt on device pointers: the product is computed by the matmul kernel.
if [ -x "$BUILD/cublaslt_client.arm64" ] && [ -f "$BUILD/kernels/matmul_macos.cubin" ]; then
    run "cublasLtMatmul on device pointers (arm64)" 0 "$BUILD/cublaslt_client.arm64"
    if [ "$HAVE_ROSETTA" = 1 ]; then
        run "cublasLtMatmul on device pointers (x86_64 under Rosetta)" 0 "$BUILD/cublaslt_client.x86_64"
    fi
else
    echo "[SKIP] cublasLtMatmul (client or kernels/matmul_macos.cubin missing)"
fi

# hybrid_ops: the unmodified Python backend over the out-of-tree library. On
# the loopback backend this proves load, init, transfers and the PTX module
# compile; the kernels themselves only run on a card.
if [ "${HAVE_HYBRID_OPS:-0}" = 1 ] && python3 -c "import numpy" 2>/dev/null \
    && "$REPO_ROOT/tools/ptxas-resolve.sh" --print-path >/dev/null 2>&1; then
    run "hybrid_ops Python backend through the adapter (host path)" 0 \
        python3 "$ADAPTER_DIR/tests/hybrid_ops_smoke.py" "$BUILD/hybrid_ops/libcuda_compute.dylib"
else
    echo "[SKIP] hybrid_ops client (not built, or numpy / verified ptxas missing)"
fi

# A cubin for an older GPU must be refused, not run.
python3 - "$VECADD_CUBIN" "$WORK/old.cubin" <<'PY'
import sys
data = bytearray(open(sys.argv[1], "rb").read())
data[49] = 61  # e_flags SM byte: pretend sm_61
open(sys.argv[2], "wb").write(data)
PY
run "pre-Blackwell cubin is refused" 1 "$BUILD/vecadd_client.arm64" "$WORK/old.cubin"

echo ""
echo "selftest: $PASSED passed, $FAILED failed (loopback backend, no GPU)"
[ "$FAILED" = 0 ]
