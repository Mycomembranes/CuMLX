# nvcc Darwin arm64e shim

This is a DALI-style native portability layer for macOS Apple Silicon. It does
not install NVIDIA's closed-source `nvcc`; instead it provides an `nvcc`
command-line-compatible driver for the subset needed by this workspace, backed
by Homebrew LLVM/Clang's CUDA frontend and local compatibility headers.

The goal is to compile CUDA device code on Darwin arm64/arm64e without Docker or
Linux emulation, following the same source-level-port approach used by the DALI
Apple Silicon patch.

## Current scope

- Maps common `nvcc` flags to LLVM 21 `clang++`.
- Defaults to device-only compilation because macOS has no NVIDIA CUDA runtime.
- Provides minimal `cuda_runtime.h` and `cuda_bf16.h` compatibility headers.
- Targets `sm_120` by default when no `-arch` flag is provided.
- Injects PTX 8.7 support for Blackwell-era shuffle intrinsics.
- Lowers the math functions used by the current kernels through inline PTX so
  the compile does not depend on NVIDIA `libdevice`.
- Lowers CUDA built-in indices (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`)
  to NVPTX special-register reads instead of unresolved global symbols.

## Usage

```sh
tools/nvcc-darwin-arm64e/bin/nvcc -arch=sm_120 -O3 -c kernels.cu \
  -o ~/clustercompute/artifacts/operonfold_nvcc_darwin_arm64e/kernels_sm120.ptx
```

To install into your shell PATH:

```sh
mkdir -p ~/.local/bin
ln -sf /Users/mukshudahamed/Documents/cuda-oxide-main/tools/nvcc-darwin-arm64e/bin/nvcc ~/.local/bin/nvcc
```

This workspace has been installed that way, so `nvcc` resolves from
`~/.local/bin/nvcc` when `~/.local/bin` is on `PATH`.

## Verified kernel compile

```sh
nvcc -arch=sm_120 -O3 -c /Users/mukshudahamed/lib/drivers/cuda/kernels/kernels.cu \
  -o /Users/mukshudahamed/clustercompute/artifacts/operonfold_nvcc_darwin_arm64e/kernels_sm120.ptx
```

The verified output is PTX 8.7 targeting `sm_120`, contains all 24 kernel
entries from the combined kernel unit, and uses `%tid`/`%ctaid` special
registers for thread/block indexing.
