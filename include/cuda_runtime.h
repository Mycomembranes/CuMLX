/*
 * SPDX-License-Identifier: MIT
 *
 * cuda_runtime.h — macOS arm64e CUDA Runtime umbrella header for the
 * cudart-shim. MLX's CUDA backend includes <cuda_runtime.h>; this pulls in the
 * runtime-API types + prototypes (cuda_runtime_api.h) that libcudart.dylib
 * implements/thunks, and the device-side launch/intrinsic helpers shared with
 * tools/nvcc-darwin-arm64e/include/cuda_runtime.h.
 *
 * Host code only needs cuda_runtime_api.h; the device helpers are guarded so a
 * pure host translation unit pays nothing for them.
 */

#ifndef CUDART_SHIM_CUDA_RUNTIME_H
#define CUDART_SHIM_CUDA_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

/* Host-facing runtime API surface (types + prototypes). */
#include "cuda_runtime_api.h"

/* CUDA vector types (char1..ulonglong4) — CCCL's tuple support (enabled once a
 * toolkit is detected) specializes std::tuple for them and assumes a real
 * <vector_types.h> was pulled. Supply the host-side definitions here so it
 * resolves; uint3/dim3 stay owned below (and in cuda_runtime_api.h). Resolved
 * from the unified toolkit-root include dir (-isystem cuda-root/include). */
#include "vector_types.h"

/* ---- device-side launch attributes + intrinsics (nvcc-darwin parity) ---- *
 * Only meaningful when compiling device code; for a host-only MLX link these
 * are inert. Mirrors tools/nvcc-darwin-arm64e/include/cuda_runtime.h.
 */
#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#endif

#ifndef CUDART_SHIM_HAS_UINT3
#define CUDART_SHIM_HAS_UINT3
typedef struct uint3 {
    unsigned int x, y, z;
} uint3;
#endif

#endif /* CUDART_SHIM_CUDA_RUNTIME_H */
