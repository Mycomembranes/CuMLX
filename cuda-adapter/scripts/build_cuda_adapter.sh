#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# build_cuda_adapter.sh - build the CUDA device adapter (host-only; touches no card).
#
#   1. cuda_adapter_server            arm64 Rust server
#   2. libcuda.dylib                  fat stand-in (x86_64 + arm64) exporting the
#                                     full CUDA 10.1 Driver API symbol list
#   3. test clients                   C (both slices) and Fortran (arm64)
#      libcudart_adapter.dylib        the few CUDA runtime calls simple clients mix in
#      hybrid_ops/libcuda_compute     optional, if ~/lib/drivers/cuda is present
#   4. self-test                      loopback backend, no GPU
#
# Usage:
#   scripts/build_cuda_adapter.sh [--no-selftest] [--install [DIR]]
#
# --install copies the stand-in library into a PRIVATE root (default
# ~/.config/cuda-oxide/cuda-adapter-root). NVIDIA's installed files under
# /Library/Frameworks/CUDA.framework and /usr/local/cuda are never written.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ADAPTER_DIR/../.." && pwd)"
BUILD="$ADAPTER_DIR/build"
SELFTEST=1
INSTALL=0
INSTALL_DIR="$HOME/.config/cuda-oxide/cuda-adapter-root"

while [ $# -gt 0 ]; do
    case "$1" in
        --no-selftest) SELFTEST=0 ;;
        --install)
            INSTALL=1
            if [ $# -gt 1 ] && [ "${2#--}" = "$2" ]; then INSTALL_DIR="$2"; shift; fi
            ;;
        -h|--help) sed -n '5,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

log_info() { echo "[INFO] $1"; }
log_error() { echo "[ERROR] $1" >&2; }

check_prerequisites() {
    log_info "Checking prerequisites..."
    [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { log_error "this build is for macOS on Apple Silicon"; exit 1; }
    command -v clang >/dev/null || { log_error "clang not found: xcode-select --install"; exit 1; }
    command -v cargo >/dev/null || { log_error "cargo not found"; exit 1; }
    if command -v gfortran >/dev/null; then
        HAVE_FORTRAN=1
        log_info "  gfortran: $(gfortran --version | head -1)"
    else
        HAVE_FORTRAN=0
        log_info "  gfortran not found (brew install gcc): skipping the Fortran client"
    fi
    if arch -x86_64 /usr/bin/true 2>/dev/null; then
        HAVE_ROSETTA=1
    else
        HAVE_ROSETTA=0
        log_info "  Rosetta not installed: the x86_64 slice is built but cannot be self-tested"
    fi
}

build_runtime_layer() {
    log_info "  Compiling the minimal runtime layer (libcudart_adapter)..."
    for a in x86_64 arm64; do
        clang -arch "$a" -O2 -Wall -Wextra -Werror -fvisibility=hidden -dynamiclib \
            -install_name @rpath/libcudart_adapter.dylib -o "$BUILD/libcudart_adapter.$a.dylib" \
            "$ADAPTER_DIR/shim/cudart_min.c" -L"$BUILD" -lcuda
    done
    lipo -create "$BUILD/libcudart_adapter.x86_64.dylib" "$BUILD/libcudart_adapter.arm64.dylib" \
        -output "$BUILD/libcudart_adapter.dylib"
    rm -f "$BUILD/libcudart_adapter.x86_64.dylib" "$BUILD/libcudart_adapter.arm64.dylib"
}

# cublasLtMatmul that runs on the card (see shim/cublaslt_card.c), plus the
# kernel image it launches, copied next to it.
build_cublaslt() {
    log_info "  Compiling the card-backed cuBLASLt layer (libcublasLt_adapter)..."
    for a in x86_64 arm64; do
        clang -arch "$a" -O2 -Wall -Wextra -Werror -fvisibility=hidden -dynamiclib \
            -install_name @rpath/libcublasLt_adapter.dylib -o "$BUILD/libcublasLt_adapter.$a.dylib" \
            "$ADAPTER_DIR/shim/cublaslt_card.c" -L"$BUILD" -lcuda
    done
    lipo -create "$BUILD/libcublasLt_adapter.x86_64.dylib" "$BUILD/libcublasLt_adapter.arm64.dylib" \
        -output "$BUILD/libcublasLt_adapter.dylib"
    rm -f "$BUILD/libcublasLt_adapter.x86_64.dylib" "$BUILD/libcublasLt_adapter.arm64.dylib"
    local cubin="$REPO_ROOT/crates/cuda-driverkit/test_assets/cubins/matmul_macos.cubin"
    mkdir -p "$BUILD/kernels"
    if [ -f "$cubin" ]; then
        cp "$cubin" "$BUILD/kernels/matmul_macos.cubin"
    else
        log_error "  $cubin is missing: cublasLtMatmul will report an internal error until it is installed"
    fi
    for a in x86_64 arm64; do
        clang -arch "$a" -O1 -Wall -Wextra -Werror -o "$BUILD/cublaslt_client.$a" \
            "$ADAPTER_DIR/tests/cublaslt_client.c" -L"$BUILD" -lcuda -lcublasLt_adapter -Wl,-rpath,@executable_path
    done
}

# Optional client: the hybrid_ops CUDA library (~/lib/drivers/cuda/cuda_compute.c).
# Neither that source nor hybrid_ops is edited. The C file is compiled as it
# is, with the four function names that clash with the Python backend's
# declarations renamed, next to shim/hybrid_ops_wrapper_abi.c, which provides
# the interface the Python backend expects (see that file for the table).
build_hybrid_ops() {
    local src="${HYBRID_OPS_CUDA_SRC:-$HOME/lib/drivers/cuda}"
    local inc="${CUDA_ADAPTER_CUDA_INCLUDE:-$HOME/.config/cuda-oxide/cuda-root/include}"
    HAVE_HYBRID_OPS=0
    if [ ! -f "$src/cuda_compute.c" ] || [ ! -f "$inc/cuda.h" ] || [ ! -f "$inc/cuda_runtime.h" ]; then
        log_info "hybrid_ops client skipped (needs $src/cuda_compute.c and cuda.h + cuda_runtime.h in $inc)"
        return
    fi
    log_info "Building the hybrid_ops client library (out of tree)..."
    mkdir -p "$BUILD/hybrid_ops"
    # The repo's cuda.h is a Driver API subset without these two CUjit_option
    # values; they are the standard CUDA constants.
    # Two compiles: the renames must apply to the library's source only.
    clang -arch arm64 -O2 -fPIC -c -I"$inc" -I"$src" \
        '-DCU_JIT_TARGET=((CUjit_option)9)' '-DCU_JIT_OPTIMIZATION_LEVEL=((CUjit_option)7)' \
        -Dcudac_get_device_name=cudac0_get_device_name -Dcudac_get_last_error=cudac0_get_last_error \
        -Dcudac_dispatch=cudac0_dispatch -Dcudac_dispatch_sync=cudac0_dispatch_sync \
        -o "$BUILD/hybrid_ops/cuda_compute.o" "$src/cuda_compute.c"
    clang -arch arm64 -O2 -fPIC -c -Wall -Wextra -Werror \
        -o "$BUILD/hybrid_ops/wrapper_abi.o" "$ADAPTER_DIR/shim/hybrid_ops_wrapper_abi.c"
    clang -arch arm64 -dynamiclib -install_name @rpath/libcuda_compute.dylib \
        -o "$BUILD/hybrid_ops/libcuda_compute.dylib" \
        "$BUILD/hybrid_ops/cuda_compute.o" "$BUILD/hybrid_ops/wrapper_abi.o" \
        -L"$BUILD" -lcuda -lcudart_adapter -Wl,-rpath,@loader_path/..
    rm -f "$BUILD/hybrid_ops/cuda_compute.o" "$BUILD/hybrid_ops/wrapper_abi.o"
    HAVE_HYBRID_OPS=1
    log_info "  hybrid_ops/libcuda_compute.dylib: OK"
}

build_server() {
    log_info "Building cuda_adapter_server (release)..."
    (cd "$REPO_ROOT" && cargo build --release -p cuda-adapter-server)
    cp "$REPO_ROOT/target/release/cuda_adapter_server" "$BUILD/cuda_adapter_server"
}

build_shim() {
    log_info "Generating stubs for unimplemented symbols..."
    python3 "$SCRIPT_DIR/gen_unsupported.py"
    local flags="-O2 -Wall -Wextra -Werror -Wno-unused-function -fvisibility=hidden -dynamiclib"
    flags="$flags -install_name @rpath/libcuda.dylib -compatibility_version 1.1.0 -current_version 10.1.105"
    for a in x86_64 arm64; do
        log_info "  Compiling libcuda ($a)..."
        # shellcheck disable=SC2086
        clang -arch "$a" $flags -o "$BUILD/libcuda.$a.dylib" \
            "$ADAPTER_DIR/shim/cuda_fwd.c" "$ADAPTER_DIR/shim/generated/cuda_unsupported.c"
    done
    lipo -create "$BUILD/libcuda.x86_64.dylib" "$BUILD/libcuda.arm64.dylib" -output "$BUILD/libcuda.dylib"
    rm -f "$BUILD/libcuda.x86_64.dylib" "$BUILD/libcuda.arm64.dylib"
    # Framework layout, so DYLD_FRAMEWORK_PATH can stand in for CUDA.framework.
    mkdir -p "$BUILD/Frameworks/CUDA.framework/Versions/A"
    cp "$BUILD/libcuda.dylib" "$BUILD/Frameworks/CUDA.framework/Versions/A/CUDA"
    ln -sfn A "$BUILD/Frameworks/CUDA.framework/Versions/Current"
    ln -sfn Versions/Current/CUDA "$BUILD/Frameworks/CUDA.framework/CUDA"
}

build_clients() {
    for a in x86_64 arm64; do
        log_info "  Compiling test clients ($a)..."
        for t in vecadd_client adapter_probe; do
            clang -arch "$a" -O1 -Wall -Wextra -Werror -o "$BUILD/$t.$a" "$ADAPTER_DIR/tests/$t.c" \
                -L"$BUILD" -lcuda -Wl,-rpath,@executable_path
        done
    done
    if [ "$HAVE_FORTRAN" = 1 ]; then
        log_info "  Compiling the Fortran client (arm64)..."
        gfortran -O3 -static-libgfortran -static-libgcc -J "$BUILD" -o "$BUILD/fortran_vecadd" \
            "$ADAPTER_DIR/fortran/cuda_adapter_bindings.f90" "$ADAPTER_DIR/fortran/fortran_vecadd.f90" \
            -L"$BUILD" -lcuda -Wl,-rpath,@executable_path
    fi
}

verify_build() {
    log_info "Verifying build..."
    local archs
    archs="$(lipo -archs "$BUILD/libcuda.dylib")"
    case "$archs" in
        *x86_64*arm64*|*arm64*x86_64*) log_info "  libcuda.dylib: OK ($archs)" ;;
        *) log_error "  libcuda.dylib has slices '$archs', expected x86_64 and arm64"; exit 1 ;;
    esac
    for a in x86_64 arm64; do
        if nm -gU -arch "$a" "$BUILD/libcuda.dylib" | awk '{print $3}' | sed 's/^_//' | LC_ALL=C sort -u \
            | diff -q - "$ADAPTER_DIR/symbols/cuda_10_1.txt" >/dev/null; then
            log_info "  $a exports: OK (matches the CUDA 10.1 list)"
        else
            log_error "  $a exports differ from symbols/cuda_10_1.txt"
            exit 1
        fi
    done
    file "$BUILD/cuda_adapter_server" | grep -q arm64 || { log_error "  server is not arm64"; exit 1; }
    log_info "  cuda_adapter_server: OK (arm64)"
}

install_root() {
    log_info "Installing the stand-in library into $INSTALL_DIR ..."
    mkdir -p "$INSTALL_DIR/lib" "$INSTALL_DIR/bin"
    cp "$BUILD/libcuda.dylib" "$INSTALL_DIR/lib/libcuda.dylib"
    rm -rf "$INSTALL_DIR/Frameworks"
    cp -R "$BUILD/Frameworks" "$INSTALL_DIR/Frameworks"
    cp "$BUILD/cuda_adapter_server" "$BUILD/adapter_probe.x86_64" "$BUILD/adapter_probe.arm64" "$INSTALL_DIR/bin/"
    log_info "Installed. Launchers use it when CUDA_ADAPTER_ROOT=$INSTALL_DIR"
}

print_summary() {
    echo ""
    echo "=============================================="
    echo "CUDA device adapter build summary"
    echo "=============================================="
    echo "Build directory: $BUILD"
    echo ""
    echo "Run a client against the served card (attended, after the stack is up):"
    echo "  scripts/run_cuda_rosetta.sh /path/to/x86_64_cuda_app [args]"
    echo "  scripts/run_cuda_native.sh  /path/to/arm64_cuda_app  [args]"
    echo ""
    echo "Limits (see README.md):"
    echo "  - Driver API only; NVIDIA's libcudart / cuBLAS are not supported"
    echo "  - modules must carry PTX or an sm_120 cubin"
    echo "  - one client at a time; default stream; no dynamic shared memory"
    echo "=============================================="
}

main() {
    echo "========================================"
    echo "CUDA device adapter build"
    echo "========================================"
    mkdir -p "$BUILD"
    check_prerequisites
    build_server
    build_shim
    build_runtime_layer
    build_cublaslt
    build_clients
    build_hybrid_ops
    verify_build
    if [ "$SELFTEST" = 1 ]; then
        HAVE_ROSETTA="$HAVE_ROSETTA" HAVE_FORTRAN="$HAVE_FORTRAN" HAVE_HYBRID_OPS="$HAVE_HYBRID_OPS" bash "$SCRIPT_DIR/selftest.sh"
    fi
    if [ "$INSTALL" = 1 ]; then install_root; fi
    print_summary
}

main
