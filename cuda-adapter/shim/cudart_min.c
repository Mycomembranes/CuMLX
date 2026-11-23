/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cudart_min.c - the small part of the CUDA RUNTIME API (cuda*) that simple
 * mixed driver/runtime programs use, implemented on the adapter's Driver API.
 *
 * It is not a libcudart replacement: it has no kernel launch path, no
 * per-thread device state and no fat-binary registration, so nvcc-compiled
 * programs (which call __cudaRegisterFatBinary) do not work with it. It exists
 * for C code that manages memory and streams through cuda* and loads and
 * launches modules through cu*, such as lib/drivers/cuda/cuda_compute.c.
 *
 * A runtime device pointer (void *) is the driver's CUdeviceptr. Error codes
 * are returned unchanged: the runtime and driver enums agree for every value
 * the adapter produces.
 */
#include <stddef.h>

#define RT_API __attribute__((visibility("default")))

typedef int cudaError_t;
typedef unsigned long long CUdeviceptr;
typedef struct CUstream_st *cudaStream_t;

int cuMemAlloc_v2(CUdeviceptr *, size_t);
int cuMemFree_v2(CUdeviceptr);
int cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
int cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
int cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t);
int cuMemAllocHost_v2(void **, size_t);
int cuMemFreeHost(void *);
int cuMemGetInfo_v2(size_t *, size_t *);
int cuStreamCreate(cudaStream_t *, unsigned int);
int cuStreamDestroy_v2(cudaStream_t);
int cuStreamSynchronize(cudaStream_t);
int cuCtxSynchronize(void);
int cuGetErrorString(int, const char **);

enum { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };

static _Thread_local cudaError_t g_last_error;

static cudaError_t remember(cudaError_t code) {
    if (code != 0) g_last_error = code;
    return code;
}

RT_API cudaError_t cudaMalloc(void **device_ptr, size_t bytes) {
    if (!device_ptr) return remember(1);
    CUdeviceptr dptr = 0;
    cudaError_t rc = cuMemAlloc_v2(&dptr, bytes);
    if (rc == 0) *device_ptr = (void *)(size_t)dptr;
    return remember(rc);
}

RT_API cudaError_t cudaFree(void *device_ptr) {
    if (!device_ptr) return 0; /* freeing NULL is allowed */
    return remember(cuMemFree_v2((CUdeviceptr)(size_t)device_ptr));
}

RT_API cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, int kind) {
    switch (kind) {
    case cudaMemcpyHostToDevice: return remember(cuMemcpyHtoD_v2((CUdeviceptr)(size_t)dst, src, bytes));
    case cudaMemcpyDeviceToHost: return remember(cuMemcpyDtoH_v2(dst, (CUdeviceptr)(size_t)src, bytes));
    case cudaMemcpyDeviceToDevice:
        return remember(cuMemcpyDtoD_v2((CUdeviceptr)(size_t)dst, (CUdeviceptr)(size_t)src, bytes));
    case cudaMemcpyHostToHost: {
        unsigned char *d = (unsigned char *)dst;
        const unsigned char *s = (const unsigned char *)src;
        if (!d || !s) return remember(1);
        for (size_t i = 0; i < bytes; i++) d[i] = s[i];
        return 0;
    }
    default: return remember(1);
    }
}

RT_API cudaError_t cudaMallocHost(void **host_ptr, size_t bytes) { return remember(cuMemAllocHost_v2(host_ptr, bytes)); }

RT_API cudaError_t cudaFreeHost(void *host_ptr) {
    if (!host_ptr) return 0;
    return remember(cuMemFreeHost(host_ptr));
}

RT_API cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    return remember(cuMemGetInfo_v2(free_bytes, total_bytes));
}

RT_API cudaError_t cudaStreamCreate(cudaStream_t *stream) { return remember(cuStreamCreate(stream, 0)); }

RT_API cudaError_t cudaStreamDestroy(cudaStream_t stream) { return remember(cuStreamDestroy_v2(stream)); }

RT_API cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return remember(cuStreamSynchronize(stream)); }

RT_API cudaError_t cudaDeviceSynchronize(void) { return remember(cuCtxSynchronize()); }

RT_API cudaError_t cudaGetLastError(void) {
    cudaError_t code = g_last_error;
    g_last_error = 0;
    return code;
}

RT_API cudaError_t cudaPeekAtLastError(void) { return g_last_error; }

RT_API const char *cudaGetErrorString(cudaError_t code) {
    const char *text = NULL;
    if (cuGetErrorString(code, &text) != 0 || !text) return "unrecognized error code";
    return text;
}
