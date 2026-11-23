#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# run_common.sh - shared body of run_cuda_native.sh and run_cuda_rosetta.sh.
# Sourced with WANT_ARCH set to arm64 or x86_64.
#
# It keeps the application untouched and replaces only the library boundary:
#   1. checks the application is the expected architecture and that dyld will
#      honour DYLD_* for it (hardened or system binaries drop those variables);
#   2. readiness gate: a probe built for the same architecture must reach a
#      cuda_adapter_server and see a device, before the application starts;
#   3. puts the stand-in libcuda / CUDA.framework first on the search paths.
#
# It starts no server and drives no card. The adapter server is started by the
# operator (or the exact-serve stack) and its socket is named by
# CUDA_ADAPTER_SOCKET (default /tmp/cuda-adapter-<uid>.sock).
#
# Environment:
#   CUDA_ADAPTER_ROOT        directory holding libcuda.dylib (or lib/libcuda.dylib),
#                            Frameworks/ and the probes; default: ../build
#   CUDA_ADAPTER_SOCKET      adapter server socket
#   CUDA_ADAPTER_SKIP_PROBE  1 = skip the readiness gate (not recommended)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CUDA_ADAPTER_ROOT:-$SCRIPT_DIR/../build}"

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <application> [arguments...]" >&2
    exit 2
fi
APP="$1"
if [ ! -x "$APP" ]; then
    APP="$(command -v "$1" || true)"
fi
if [ -z "$APP" ] || [ ! -x "$APP" ]; then
    echo "application is not executable: $1" >&2
    exit 2
fi
shift

if [ -f "$ROOT/lib/libcuda.dylib" ]; then
    LIB_DIR="$ROOT/lib"
    PROBE="$ROOT/bin/adapter_probe.$WANT_ARCH"
elif [ -f "$ROOT/libcuda.dylib" ]; then
    LIB_DIR="$ROOT"
    PROBE="$ROOT/adapter_probe.$WANT_ARCH"
else
    echo "stand-in libcuda.dylib not found under $ROOT" >&2
    echo "Build it first with: portkit/cuda-adapter/scripts/build_cuda_adapter.sh" >&2
    exit 2
fi
LIB_DIR="$(cd "$LIB_DIR" && pwd)"
FRAMEWORK_DIR="$(cd "$ROOT/Frameworks" 2>/dev/null && pwd || true)"

# 1. Architecture and DYLD_* admission. Scripts are let through: the
#    interpreter decides what it loads.
if file -b "$APP" | grep -q "Mach-O"; then
    if ! lipo -archs "$APP" 2>/dev/null | tr ' ' '\n' | grep -qx "$WANT_ARCH"; then
        echo "$APP has no $WANT_ARCH slice (it has: $(lipo -archs "$APP" 2>/dev/null))" >&2
        exit 2
    fi
    case "$APP" in
        /System/*|/usr/bin/*|/bin/*|/sbin/*|/usr/sbin/*)
            echo "$APP is a system binary: macOS strips DYLD_* for it, so the adapter cannot be loaded" >&2
            exit 2 ;;
    esac
    signature="$(codesign -d --verbose=4 "$APP" 2>&1 || true)"
    if echo "$signature" | grep -q "flags=.*runtime"; then
        if ! codesign -d --entitlements - "$APP" 2>/dev/null | grep -q "allow-dyld-environment-variables"; then
            echo "$APP is signed with the hardened runtime and without" >&2
            echo "com.apple.security.cs.allow-dyld-environment-variables: macOS will ignore DYLD_* for it." >&2
            echo "Re-sign your own copy ad hoc (codesign --force --sign - <copy>) and run that copy." >&2
            exit 2
        fi
    fi
fi

export DYLD_LIBRARY_PATH="$LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
if [ -n "$FRAMEWORK_DIR" ]; then
    export DYLD_FRAMEWORK_PATH="$FRAMEWORK_DIR${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}"
fi

# 2. Readiness gate.
if [ "${CUDA_ADAPTER_SKIP_PROBE:-0}" != "1" ]; then
    if [ ! -x "$PROBE" ]; then
        echo "readiness probe not found: $PROBE" >&2
        exit 2
    fi
    if ! "$PROBE"; then
        echo "CUDA adapter readiness gate failed: no adapter server is answering." >&2
        echo "Start cuda_adapter_server (attended, with the stack served) or set CUDA_ADAPTER_SOCKET." >&2
        exit 3
    fi
fi

# 3. Hand off.
if [ "$WANT_ARCH" = "x86_64" ]; then
    exec arch -x86_64 "$APP" "$@"
fi
exec "$APP" "$@"
