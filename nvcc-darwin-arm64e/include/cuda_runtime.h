#pragma once

#include <stdint.h>
#include <stddef.h>
#include <math.h>

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

// CUDA vector types (char1..ulonglong4) — CCCL's tuple support references them
// once a toolkit is detected. uint3 + dim3 stay below (this header owns them).
#include "vector_types.h"

typedef struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} uint3;

struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;

    __host__ __device__ constexpr dim3(
        unsigned int vx = 1,
        unsigned int vy = 1,
        unsigned int vz = 1
    ) : x(vx), y(vy), z(vz) {}
};

typedef void *cudaStream_t;
typedef int cudaError_t;

extern "C" __host__ cudaError_t cudaConfigureCall(
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem = 0,
    cudaStream_t stream = 0
);
extern "C" __host__ cudaError_t cudaSetupArgument(
    const void *arg,
    size_t size,
    size_t offset
);
extern "C" __host__ cudaError_t cudaLaunch(const void *func);

extern "C" __device__ int __nvvm_read_ptx_sreg_tid_x();
extern "C" __device__ int __nvvm_read_ptx_sreg_tid_y();
extern "C" __device__ int __nvvm_read_ptx_sreg_tid_z();
extern "C" __device__ int __nvvm_read_ptx_sreg_ctaid_x();
extern "C" __device__ int __nvvm_read_ptx_sreg_ctaid_y();
extern "C" __device__ int __nvvm_read_ptx_sreg_ctaid_z();
extern "C" __device__ int __nvvm_read_ptx_sreg_ntid_x();
extern "C" __device__ int __nvvm_read_ptx_sreg_ntid_y();
extern "C" __device__ int __nvvm_read_ptx_sreg_ntid_z();
extern "C" __device__ int __nvvm_read_ptx_sreg_nctaid_x();
extern "C" __device__ int __nvvm_read_ptx_sreg_nctaid_y();
extern "C" __device__ int __nvvm_read_ptx_sreg_nctaid_z();

__device__ __forceinline__ uint3 __nvcc_darwin_thread_idx() {
    uint3 out;
    out.x = (unsigned int)__nvvm_read_ptx_sreg_tid_x();
    out.y = (unsigned int)__nvvm_read_ptx_sreg_tid_y();
    out.z = (unsigned int)__nvvm_read_ptx_sreg_tid_z();
    return out;
}

__device__ __forceinline__ uint3 __nvcc_darwin_block_idx() {
    uint3 out;
    out.x = (unsigned int)__nvvm_read_ptx_sreg_ctaid_x();
    out.y = (unsigned int)__nvvm_read_ptx_sreg_ctaid_y();
    out.z = (unsigned int)__nvvm_read_ptx_sreg_ctaid_z();
    return out;
}

__device__ __forceinline__ uint3 __nvcc_darwin_block_dim() {
    uint3 out;
    out.x = (unsigned int)__nvvm_read_ptx_sreg_ntid_x();
    out.y = (unsigned int)__nvvm_read_ptx_sreg_ntid_y();
    out.z = (unsigned int)__nvvm_read_ptx_sreg_ntid_z();
    return out;
}

__device__ __forceinline__ uint3 __nvcc_darwin_grid_dim() {
    uint3 out;
    out.x = (unsigned int)__nvvm_read_ptx_sreg_nctaid_x();
    out.y = (unsigned int)__nvvm_read_ptx_sreg_nctaid_y();
    out.z = (unsigned int)__nvvm_read_ptx_sreg_nctaid_z();
    return out;
}

#define threadIdx (__nvcc_darwin_thread_idx())
#define blockIdx (__nvcc_darwin_block_idx())
#define blockDim (__nvcc_darwin_block_dim())
#define gridDim (__nvcc_darwin_grid_dim())

extern "C" __device__ void __syncthreads();
extern "C" __device__ int __nvvm_shfl_sync_down_i32(
    unsigned int mask,
    int value,
    int offset,
    int width
);

__device__ __forceinline__ unsigned int __shfl_down_sync(
    unsigned int mask,
    unsigned int value,
    unsigned int delta,
    int width = 32
) {
    return (unsigned int)__nvvm_shfl_sync_down_i32(mask, (int)value, (int)delta, width);
}

__device__ __forceinline__ float __nvcc_darwin_sqrtf(float value) {
    float out;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}

__device__ __forceinline__ float __nvcc_darwin_expf(float value) {
    float out;
    const float scaled = value * 1.4426950408889634f;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(scaled));
    return out;
}

__device__ __forceinline__ float __nvcc_darwin_logf(float value) {
    float out;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out * 0.6931471805599453f;
}

__device__ __forceinline__ float __nvcc_darwin_fabsf(float value) {
    return value < 0.0f ? -value : value;
}

__device__ __forceinline__ float __nvcc_darwin_fmaxf(float a, float b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float __nvcc_darwin_sinf(float value) {
    float out;
    asm volatile("sin.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}

__device__ __forceinline__ float __nvcc_darwin_cosf(float value) {
    float out;
    asm volatile("cos.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}

#define sqrtf __nvcc_darwin_sqrtf
#define expf __nvcc_darwin_expf
#define logf __nvcc_darwin_logf
#define fabsf __nvcc_darwin_fabsf
#define fmaxf __nvcc_darwin_fmaxf
#define sinf __nvcc_darwin_sinf
#define cosf __nvcc_darwin_cosf

__device__ __forceinline__ int __shfl_down_sync(
    unsigned int mask,
    int value,
    unsigned int delta,
    int width = 32
) {
    return __nvvm_shfl_sync_down_i32(mask, value, (int)delta, width);
}

__device__ __forceinline__ float __shfl_down_sync(
    unsigned int mask,
    float value,
    unsigned int delta,
    int width = 32
) {
    union {
        float f;
        int i;
    } in, out;
    in.f = value;
    out.i = __nvvm_shfl_sync_down_i32(mask, in.i, (int)delta, width);
    return out.f;
}
