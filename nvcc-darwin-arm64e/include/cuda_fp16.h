// cuda_fp16.h — minimal macOS arm64 HOST-compile shim for MLX's CUDA backend.
//
// Mirrors the proven cuda_bf16.h shim. The DEVICE compile path uses NVIDIA's
// real cuda_fp16.h (via --cuda-path=$CUDA_OXIDE_CROSS_CUDA_ROOT), so this shim
// is ONLY seen by host .cpp TUs (Apple clang, -isystem cuda-root/include). It
// exists because the real redist cuda_fp16.h drags in vector_types.h /
// driver_types.h, which REDEFINE the shim's inline dim3/uint3/cudaError/... .
//
// Host code needs only the __half / __half2 TYPES plus the float<->half
// conversions (the __h* arithmetic intrinsics MLX uses live in dependent
// template branches that are instantiated for device only — exactly as for
// __nv_bfloat16). The conversions are `static inline` (host-callable, no
// dependency on the __device__/__forceinline__ CUDA macros — self-contained even
// if this header is pulled before cuda_runtime.h; never executed on the GPU,
// real bit-exact compute is sp3-canonical), with bit-compatible 16-bit storage
// so the host/device ABI across a kernel launch matches the real type.
#pragma once

#include <stdint.h>

struct __half {
    uint16_t __x;
};

// NVIDIA's real cuda_fp16.h exposes the short alias; cudnn-frontend uses it
// (e.g. std::variant<..., half, ...> in graph_properties.h).
typedef __half half;

// 4-byte packed pair, ABI-compatible with the real __half2 (2x 16-bit).
struct __half2 {
    __half x;
    __half y;
};

static inline float __half2float(__half h) {
    uint32_t x = h.__x;
    uint32_t sign = (x & 0x8000u) << 16;
    uint32_t exp = (x >> 10) & 0x1Fu;
    uint32_t mant = x & 0x3FFu;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;  // +/-0
        } else {
            // subnormal half -> normalized float
            exp = 1u;
            while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } o;
    o.u = out;
    return o.f;
}

static inline __half __float2half_rn(float value) {
    union { float f; uint32_t u; } in;
    in.f = value;
    uint32_t x = in.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    __half out;
    if (exp <= 0) {
        out.__x = (uint16_t)sign;                       // underflow -> +/-0
    } else if (exp >= 0x1F) {
        out.__x = (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));  // overflow -> inf/nan
    } else {
        out.__x = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    }
    return out;
}

static inline __half __float2half(float value) {
    return __float2half_rn(value);
}
