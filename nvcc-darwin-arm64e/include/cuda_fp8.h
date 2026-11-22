// cuda_fp8.h — minimal macOS arm64 HOST-compile shim for MLX's CUDA backend.
//
// Companion to the cuda_fp16.h / cuda_bf16.h shims (see those for the rationale:
// the real redist cuda_fp8.h drags in vector_types.h / driver_types.h that clash
// with the shim's inline runtime types; the device path uses the real header via
// --cuda-path). MLX host code only names the FP8 storage TYPE (`__nv_fp8_e4m3`);
// the conversions/intrinsics live in device-only template branches. 8-bit
// storage, ABI-compatible with the real type across a kernel launch.
#pragma once

#include <stdint.h>

typedef enum {
    __NV_E4M3 = 0,
    __NV_E5M2 = 1,
} __nv_fp8_interpretation_t;

struct __nv_fp8_e4m3 {
    uint8_t __x;
};

struct __nv_fp8_e5m2 {
    uint8_t __x;
};
