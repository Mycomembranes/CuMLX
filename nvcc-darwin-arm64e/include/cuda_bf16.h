#pragma once

#include <stdint.h>

struct __nv_bfloat16 {
    uint16_t __x;
};

// NVIDIA's real cuda_bf16.h exposes the short alias; cudnn-frontend uses it
// (e.g. std::variant<..., nv_bfloat16> in graph_properties.h).
typedef __nv_bfloat16 nv_bfloat16;

__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 value) {
    union {
        uint32_t u;
        float f;
    } out;
    out.u = ((uint32_t)value.__x) << 16;
    return out.f;
}

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float value) {
    union {
        float f;
        uint32_t u;
    } in;
    in.f = value;
    __nv_bfloat16 out;
    out.__x = (uint16_t)((in.u + 0x8000u) >> 16);
    return out;
}

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float value) {
    return __float2bfloat16_rn(value);
}
