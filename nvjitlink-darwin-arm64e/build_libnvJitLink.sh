#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${ROOT}/target/release"
LIB_REAL="${TARGET}/libnvjitlink_darwin_arm64e.dylib"
LIB_ALIAS="${TARGET}/libnvJitLink.dylib"

cargo build --release -p nvjitlink-darwin-arm64e

if [[ ! -f "${LIB_REAL}" ]]; then
    echo "[build_libnvJitLink] missing built dylib: ${LIB_REAL}" >&2
    exit 1
fi

ln -sf "${LIB_REAL}" "${LIB_ALIAS}"
echo "[build_libnvJitLink] built: ${LIB_REAL}"
echo "[build_libnvJitLink] alias: ${LIB_ALIAS} -> ${LIB_REAL}"
