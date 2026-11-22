#!/usr/bin/env python3
"""Compile every kernel in cuda-oxide-main to PTX using the in-tree nvcc shim.

Outputs:
  target/nvcc-arm64e/cu/*.ptx
  target/nvcc-arm64e/rust-kernels/*.ptx
  target/nvcc-arm64e/manifest.json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/mukshudahamed/Documents/cuda-oxide-main")
NVCC = ROOT / "tools/nvcc-darwin-arm64e/bin/nvcc"
SHIM_INCLUDE = ROOT / "tools/nvcc-darwin-arm64e/include"
OUT_DIR = ROOT / "target/nvcc-arm64e"
CU_OUT = OUT_DIR / "cu"
RUST_OUT = OUT_DIR / "rust-kernels"
LOG_DIR = OUT_DIR / "logs"
MANIFEST = OUT_DIR / "manifest.json"

ARCH = "sm_120"
HOST_TRIPLE = "arm64-apple-darwin"   # observed from nvcc -v
TIMEOUT = 120

CU_FILES = [
    "crates/rustc-codegen-cuda/examples/mathdx_ffi_test/extern-libs/test_separate.cu",
    "crates/rustc-codegen-cuda/examples/mathdx_ffi_test/extern-libs/cuda_test_kernels.cu",
    "crates/rustc-codegen-cuda/examples/mathdx_ffi_test/extern-libs/cublasdx_wrappers.cu",
    "crates/rustc-codegen-cuda/examples/mathdx_ffi_test/extern-libs/cufftdx_wrappers.cu",
    "crates/rustc-codegen-cuda/examples/mathdx_ffi_test/extern-libs/cufftdx_wrappers_funcs.cu",
    "crates/rustc-codegen-cuda/examples/device_ffi_test/tools/launch_cubin.cu",
    "crates/rustc-codegen-cuda/examples/device_ffi_test/extern-libs/external_device_funcs.cu",
    "crates/rustc-codegen-cuda/examples/device_ffi_test/extern-libs/cccl_wrappers.cu",
    "crates/rustc-codegen-cuda/examples/cpp_consumes_rust_device/cuda-caller/caller_kernel.cu",
    "crates/rustc-codegen-cuda/examples/cpp_consumes_rust_device/cuda-caller/test_runner.cu",
]

# Rust #[kernel] examples called out in the task spec.
RUST_KERNELS = [
    "vecadd_macos",
    "host_closure",
    "tcgen05",
    "f16_stress",
    "sharedmem",
    "mcast_barrier_test",
    "dynamic_smem",
    "gemm_sol",
    "tma_copy",
    "standalone_device_fn",
    "compiler_features",
]

PTX_VERSION_RE = re.compile(r"^\.version\s+(\S+)", re.MULTILINE)
PTX_ENTRY_RE = re.compile(r"^\s*\.entry\b", re.MULTILINE)
PTX_VISIBLE_ENTRY_RE = re.compile(r"^\s*\.visible\s+\.entry\b", re.MULTILINE)


def first5(stderr: str) -> str:
    if not stderr:
        return ""
    return "\n".join(stderr.splitlines()[:5])


def categorize_failure(stderr: str) -> str:
    """Return 'missing_dep' for header-not-found errors, 'compile_failed' otherwise."""
    if not stderr:
        return "compile_failed"
    lower = stderr.lower()
    # Match clang's 'fatal error: 'foo.h' file not found'
    if "file not found" in lower or "no such file" in lower:
        return "missing_dep"
    return "compile_failed"


def extract_ptx_meta(ptx_path: Path) -> tuple[str | None, int]:
    try:
        text = ptx_path.read_text(errors="replace")
    except OSError:
        return None, 0
    ver_match = PTX_VERSION_RE.search(text)
    version = ver_match.group(1) if ver_match else None
    # Count entries — prefer .visible .entry but fall back to any .entry.
    n_entries = len(PTX_VISIBLE_ENTRY_RE.findall(text))
    if n_entries == 0:
        n_entries = len(PTX_ENTRY_RE.findall(text))
    return version, n_entries


def compile_cu(rel_src: str) -> dict:
    src = ROOT / rel_src
    name = src.stem
    out = CU_OUT / f"{name}.ptx"
    log = LOG_DIR / f"{name}.cu.log"
    src_dir = src.parent

    # The shim emits PTX via clang's `-S` regardless of `-c` / `-ptx`. NVIDIA's
    # nvcc auto-includes cuda_runtime.h; the shim does not, so we add
    # `-include cuda_runtime.h` ourselves.
    base = [
        str(NVCC), f"-arch={ARCH}", "-O3", "-c",
        "-include", "cuda_runtime.h",
    ]
    attempts = [
        base + ["-o", str(out), str(src)],
        base + [
            "-I", str(src_dir),
            "-I", str(SHIM_INCLUDE),
            "-o", str(out), str(src),
        ],
    ]

    last_stderr = ""
    last_cmd: list[str] = []
    for cmd in attempts:
        last_cmd = cmd
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired as e:
            last_stderr = f"timeout after {TIMEOUT}s: {e}"
            continue

        last_stderr = (r.stderr or "") + (r.stdout or "")
        if r.returncode == 0 and out.exists():
            log.write_text(last_stderr)
            version, n_entries = extract_ptx_meta(out)
            return {
                "source": str(src),
                "kind": "cu",
                "output": str(out),
                "arch": ARCH,
                "host_triple": HOST_TRIPLE,
                "ptx_version": version,
                "kernel_count": n_entries,
                "bytes": out.stat().st_size,
                "status": "ok",
                "error": None,
                "command": " ".join(cmd),
            }

    log.write_text(last_stderr)
    status = categorize_failure(last_stderr)
    if out.exists():
        out.unlink()
    return {
        "source": str(src),
        "kind": "cu",
        "output": None,
        "arch": ARCH,
        "host_triple": HOST_TRIPLE,
        "ptx_version": None,
        "kernel_count": 0,
        "bytes": 0,
        "status": status,
        "error": first5(last_stderr),
        "command": " ".join(last_cmd),
    }


def find_workspace_backend() -> Path | None:
    candidates = [
        ROOT / "target/codex-cuda-oxide-target/debug/librustc_codegen_cuda.dylib",
        ROOT / "target/codex-cuda-oxide-target/release/librustc_codegen_cuda.dylib",
        ROOT / "target/debug/librustc_codegen_cuda.dylib",
        ROOT / "target/release/librustc_codegen_cuda.dylib",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_cargo_oxide() -> Path | None:
    for c in [
        ROOT / "target/release/cargo-oxide",
        ROOT / "target/debug/cargo-oxide",
    ]:
        if c.exists():
            return c
    return None


def compile_rust_kernel(name: str, backend_so: Path | None, cargo_oxide: Path | None) -> dict:
    example_dir = ROOT / "crates/rustc-codegen-cuda/examples" / name
    cargo_toml = example_dir / "Cargo.toml"
    log = LOG_DIR / f"{name}.rust.log"

    if not cargo_toml.is_file():
        log.write_text(f"missing Cargo.toml at {cargo_toml}\n")
        return {
            "source": str(example_dir),
            "kind": "rust-kernel",
            "output": None,
            "arch": ARCH,
            "host_triple": HOST_TRIPLE,
            "ptx_version": None,
            "kernel_count": 0,
            "bytes": 0,
            "status": "skipped",
            "error": f"no Cargo.toml at {cargo_toml}",
            "command": "",
        }

    if backend_so is None:
        log.write_text("librustc_codegen_cuda.dylib not found in target/")
        return {
            "source": str(example_dir),
            "kind": "rust-kernel",
            "output": None,
            "arch": ARCH,
            "host_triple": HOST_TRIPLE,
            "ptx_version": None,
            "kernel_count": 0,
            "bytes": 0,
            "status": "missing_dep",
            "error": "rustc-codegen-cuda backend .dylib not built; cannot drive cargo path",
            "command": "",
        }

    # Use cargo build with the backend (no run, since hardware may be missing).
    rustflags = (
        f"-Z codegen-backend={backend_so} "
        f"-C opt-level=3 -C debug-assertions=off "
        f"-Z mir-enable-passes=-JumpThreading"
    )
    env = os.environ.copy()
    env["RUSTFLAGS"] = rustflags
    env["CUDA_OXIDE_TARGET"] = ARCH

    cmd = ["cargo", "build", "--release"]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(example_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        log.write_text(f"timeout after {TIMEOUT}s\n{e}")
        return {
            "source": str(example_dir),
            "kind": "rust-kernel",
            "output": None,
            "arch": ARCH,
            "host_triple": HOST_TRIPLE,
            "ptx_version": None,
            "kernel_count": 0,
            "bytes": 0,
            "status": "compile_failed",
            "error": f"timeout after {TIMEOUT}s",
            "command": " ".join(cmd),
        }

    combined = (r.stderr or "") + "\n" + (r.stdout or "")
    log.write_text(combined)

    # Look for produced PTX in example dir or inside its target.
    ptx_paths = list(example_dir.glob("*.ptx"))
    # Prefer the one named after the example.
    ptx_paths.sort(key=lambda p: 0 if p.stem == name else 1)
    if ptx_paths and r.returncode == 0:
        src_ptx = ptx_paths[0]
        out = RUST_OUT / f"{name}.ptx"
        shutil.copy2(src_ptx, out)
        version, n_entries = extract_ptx_meta(out)
        return {
            "source": str(example_dir),
            "kind": "rust-kernel",
            "output": str(out),
            "arch": ARCH,
            "host_triple": HOST_TRIPLE,
            "ptx_version": version,
            "kernel_count": n_entries,
            "bytes": out.stat().st_size,
            "status": "ok",
            "error": None,
            "command": " ".join(cmd),
        }

    # No PTX produced. Build either failed or backend didn't emit.
    return {
        "source": str(example_dir),
        "kind": "rust-kernel",
        "output": None,
        "arch": ARCH,
        "host_triple": HOST_TRIPLE,
        "ptx_version": None,
        "kernel_count": 0,
        "bytes": 0,
        "status": "compile_failed",
        "error": first5(combined),
        "command": " ".join(cmd),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CU_OUT.mkdir(parents=True, exist_ok=True)
    RUST_OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[nvcc-arm64e] using nvcc shim at {NVCC}")
    nvcc_v = subprocess.run(
        [str(NVCC), "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=10,
    ).stdout.splitlines()[:5]
    print("[nvcc-arm64e] nvcc -v head -5:")
    for line in nvcc_v:
        print(f"  {line}")

    entries: list[dict] = []

    # 1) .cu in parallel.
    print(f"\n[nvcc-arm64e] compiling {len(CU_FILES)} .cu files in parallel ...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(compile_cu, rel): rel for rel in CU_FILES}
        for fut in as_completed(futures):
            res = fut.result()
            entries.append(res)
            tag = res["status"]
            print(f"  [{tag:>14}] {Path(res['source']).name}")

    # 2) Rust #[kernel] examples — sequentially, since cargo locks the target dir.
    backend_so = find_workspace_backend()
    cargo_oxide = find_cargo_oxide()
    print(f"\n[nvcc-arm64e] compiling {len(RUST_KERNELS)} Rust #[kernel] examples ...")
    print(f"  backend_so = {backend_so}")
    print(f"  cargo-oxide = {cargo_oxide}")

    for name in RUST_KERNELS:
        res = compile_rust_kernel(name, backend_so, cargo_oxide)
        entries.append(res)
        print(f"  [{res['status']:>14}] {name}")

    MANIFEST.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "nvcc_v_head5": nvcc_v,
        "entries": entries,
    }, indent=2))
    print(f"\n[nvcc-arm64e] manifest written to {MANIFEST}")

    # Summary
    n = len(entries)
    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    print("\n=== SUMMARY ===")
    print(f"attempted: {n}")
    for k, v in sorted(by_status.items()):
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
