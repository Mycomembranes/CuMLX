/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cublaslt_client.c - calls cuBLASLt the way MLX's CUDA backend does (create
 * handle, descriptor, three layouts, preference, heuristic, matmul) on device
 * pointers, and checks the product against a host triple loop.
 *
 * Inputs are small integers stored as float, so every partial sum is exact in
 * single precision and the comparison is bit for bit whatever order the
 * kernel adds in. Also checks that unsupported configurations are refused.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
int cuInit(unsigned);
int cuDeviceGet(int *, int);
int cuCtxCreate_v2(CUcontext *, unsigned, int);
int cuCtxDestroy_v2(CUcontext);
int cuMemAlloc_v2(CUdeviceptr *, size_t);
int cuMemFree_v2(CUdeviceptr);
int cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
int cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);

typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;
typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    int state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;
int cublasLtCreate(void **);
int cublasLtDestroy(void *);
int cublasLtMatmulDescCreate(void **, int, int);
int cublasLtMatmulDescDestroy(void *);
int cublasLtMatmulDescSetAttribute(void *, int, const void *, size_t);
int cublasLtMatrixLayoutCreate(void **, int, uint64_t, uint64_t, int64_t);
int cublasLtMatrixLayoutDestroy(void *);
int cublasLtMatmulPreferenceCreate(void **);
int cublasLtMatmulPreferenceDestroy(void *);
int cublasLtMatmulAlgoGetHeuristic(void *, void *, void *, void *, void *, void *, void *, int,
                                   cublasLtMatmulHeuristicResult_t *, int *);
int cublasLtMatmul(void *, void *, const void *, const void *, void *, const void *, void *, const void *, const void *,
                   void *, void *, void *, const cublasLtMatmulAlgo_t *, void *, size_t, void *);

enum { CUDA_R_32F = 0, CUBLAS_COMPUTE_32F = 68, NOT_SUPPORTED = 15, TRANSA = 3, OP_T = 1 };

#define CHECK(call)                                                   \
    do {                                                              \
        int rc_ = (call);                                             \
        if (rc_ != 0) {                                               \
            fprintf(stderr, "FAIL %s -> %d\n", #call, rc_);           \
            return 1;                                                 \
        }                                                             \
    } while (0)

static int failed;
#define EXPECT(cond)                                                  \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL expected %s\n", #cond);             \
            failed = 1;                                               \
        }                                                             \
    } while (0)

int main(void) {
    /* Not multiples of 16 on purpose: the kernel's 16x16 blocks overhang. */
    enum { M = 37, K = 21, N = 50 };
    static float a[M * K], b[K * N], d[M * N], want[M * N];
    /* Column-major: element (row, col) of an R-row matrix is [row + col * R]. */
    for (int col = 0; col < K; col++)
        for (int row = 0; row < M; row++) a[row + col * M] = (float)((row * 3 + col * 5) % 7 - 3);
    for (int col = 0; col < N; col++)
        for (int row = 0; row < K; row++) b[row + col * K] = (float)((row * 2 + col) % 5 - 2);
    for (int col = 0; col < N; col++)
        for (int row = 0; row < M; row++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) sum += a[row + i * M] * b[i + col * K];
            want[row + col * M] = sum;
        }

    int dev;
    CUcontext ctx;
    CUdeviceptr da, db, dd;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate_v2(&ctx, 0, dev));
    CHECK(cuMemAlloc_v2(&da, sizeof(a)));
    CHECK(cuMemAlloc_v2(&db, sizeof(b)));
    CHECK(cuMemAlloc_v2(&dd, sizeof(d)));
    CHECK(cuMemcpyHtoD_v2(da, a, sizeof(a)));
    CHECK(cuMemcpyHtoD_v2(db, b, sizeof(b)));

    void *handle, *desc, *la, *lb, *ld, *pref;
    CHECK(cublasLtCreate(&handle));
    CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, M, K, M));
    CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, N, K));
    CHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, M, N, M));
    CHECK(cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulHeuristicResult_t heuristic;
    int found = 0;
    CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, ld, ld, pref, 1, &heuristic, &found));
    EXPECT(found == 1 && heuristic.state == 0 && heuristic.workspaceSize == 0);

    float alpha = 1.0f, beta = 0.0f;
    CHECK(cublasLtMatmul(handle, desc, &alpha, (void *)(uintptr_t)da, la, (void *)(uintptr_t)db, lb, &beta,
                         (void *)(uintptr_t)dd, ld, (void *)(uintptr_t)dd, ld, &heuristic.algo, NULL, 0, NULL));
    CHECK(cuMemcpyDtoH_v2(d, dd, sizeof(d)));
    int wrong = 0;
    for (int i = 0; i < M * N; i++) wrong += d[i] != want[i];
    if (wrong) {
        fprintf(stderr, "FAIL %d of %d product elements differ (first: got %g, want %g)\n", wrong, M * N, d[0], want[0]);
        failed = 1;
    }

    /* Refusals: nothing may be computed for a configuration that is not modelled. */
    float two = 2.0f, one = 1.0f;
    EXPECT(cublasLtMatmul(handle, desc, &two, (void *)(uintptr_t)da, la, (void *)(uintptr_t)db, lb, &beta,
                          (void *)(uintptr_t)dd, ld, (void *)(uintptr_t)dd, ld, NULL, NULL, 0, NULL) == NOT_SUPPORTED);
    EXPECT(cublasLtMatmul(handle, desc, &alpha, (void *)(uintptr_t)da, la, (void *)(uintptr_t)db, lb, &one,
                          (void *)(uintptr_t)dd, ld, (void *)(uintptr_t)dd, ld, NULL, NULL, 0, NULL) == NOT_SUPPORTED);
    EXPECT(cublasLtMatmul(handle, desc, &alpha, (void *)(uintptr_t)da, la, (void *)(uintptr_t)db, lb, &beta,
                          (void *)(uintptr_t)da, ld, (void *)(uintptr_t)da, ld, NULL, NULL, 0, NULL) == NOT_SUPPORTED);
    void *padded;
    CHECK(cublasLtMatrixLayoutCreate(&padded, CUDA_R_32F, M, K, M + 3)); /* leading dimension > rows */
    EXPECT(cublasLtMatmul(handle, desc, &alpha, (void *)(uintptr_t)da, padded, (void *)(uintptr_t)db, lb, &beta,
                          (void *)(uintptr_t)dd, ld, (void *)(uintptr_t)dd, ld, NULL, NULL, 0, NULL) == NOT_SUPPORTED);
    int transpose = OP_T;
    CHECK(cublasLtMatmulDescSetAttribute(desc, TRANSA, &transpose, sizeof(transpose)));
    EXPECT(cublasLtMatmul(handle, desc, &alpha, (void *)(uintptr_t)da, la, (void *)(uintptr_t)db, lb, &beta,
                          (void *)(uintptr_t)dd, ld, (void *)(uintptr_t)dd, ld, NULL, NULL, 0, NULL) == NOT_SUPPORTED);
    /* The refused calls must not have touched the result. */
    static float after[M * N];
    CHECK(cuMemcpyDtoH_v2(after, dd, sizeof(after)));
    EXPECT(memcmp(after, d, sizeof(d)) == 0);

    CHECK(cublasLtMatrixLayoutDestroy(padded));
    CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CHECK(cublasLtMatrixLayoutDestroy(la));
    CHECK(cublasLtMatrixLayoutDestroy(lb));
    CHECK(cublasLtMatrixLayoutDestroy(ld));
    CHECK(cublasLtMatmulDescDestroy(desc));
    CHECK(cublasLtDestroy(handle));
    CHECK(cuMemFree_v2(da));
    CHECK(cuMemFree_v2(db));
    CHECK(cuMemFree_v2(dd));
    CHECK(cuCtxDestroy_v2(ctx));
    if (failed) {
        fprintf(stderr, "cublaslt_client: FAILED\n");
        return 1;
    }
    printf("cublaslt_client: PASS (%dx%d * %dx%d on device pointers, bit-exact)\n", M, K, K, N);
    return 0;
}
