/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * launch_bench.c - host wall-clock cost of one kernel launch through the
 * adapter. usage: launch_bench <vecadd cubin> [launches, default 200]
 *
 * It times the client-observed round trip of cuLaunchKernel on a 1024-element
 * vecadd. That is host latency (socket, adapter, launch server, submit and
 * completion wait), NOT GPU execution time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned long long CUdeviceptr;
int cuInit(unsigned);
int cuDeviceGet(int *, int);
int cuCtxCreate_v2(void **, unsigned, int);
int cuCtxDestroy_v2(void *);
int cuMemAlloc_v2(CUdeviceptr *, size_t);
int cuMemFree_v2(CUdeviceptr);
int cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
int cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
int cuModuleLoad(void **, const char *);
int cuModuleGetFunction(void **, void *, const char *);
int cuLaunchKernel(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **);

#define CHECK(call)                                              \
    do {                                                         \
        int rc_ = (call);                                        \
        if (rc_ != 0) {                                          \
            fprintf(stderr, "FAIL %s -> %d\n", #call, rc_);      \
            return 1;                                            \
        }                                                        \
    } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static int by_value(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vecadd cubin> [launches]\n", argv[0]);
        return 2;
    }
    int launches = argc > 2 ? atoi(argv[2]) : 200;
    if (launches < 1 || launches > 100000) return 2;
    enum { N = 1024 };
    static float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }
    int dev;
    void *ctx, *module, *fn;
    CUdeviceptr da, db, dc;
    unsigned n = N;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate_v2(&ctx, 0, dev));
    CHECK(cuModuleLoad(&module, argv[1]));
    CHECK(cuModuleGetFunction(&fn, module, "vecadd_macos"));
    CHECK(cuMemAlloc_v2(&da, sizeof(a)));
    CHECK(cuMemAlloc_v2(&db, sizeof(b)));
    CHECK(cuMemAlloc_v2(&dc, sizeof(c)));
    CHECK(cuMemcpyHtoD_v2(da, a, sizeof(a)));
    CHECK(cuMemcpyHtoD_v2(db, b, sizeof(b)));
    void *params[] = {&da, &db, &dc, &n};
    double *each = malloc(sizeof(double) * (size_t)launches);
    if (!each) return 2;
    double start = now_ms();
    for (int i = 0; i < launches; i++) {
        double t0 = now_ms();
        CHECK(cuLaunchKernel(fn, 4, 1, 1, 256, 1, 1, 0, NULL, params, NULL));
        each[i] = now_ms() - t0;
    }
    double total = now_ms() - start;
    CHECK(cuMemcpyDtoH_v2(c, dc, sizeof(c)));
    for (int i = 0; i < N; i++)
        if (c[i] != a[i] + b[i]) {
            fprintf(stderr, "FAIL wrong result at %d\n", i);
            return 1;
        }
    qsort(each, (size_t)launches, sizeof(double), by_value);
    printf("%d launches in %.1f ms: mean %.3f ms, median %.3f ms, min %.3f ms, p95 %.3f ms, max %.3f ms (host round trip)\n",
           launches, total, total / launches, each[launches / 2], each[0], each[(launches * 95) / 100], each[launches - 1]);
    CHECK(cuMemFree_v2(da));
    CHECK(cuMemFree_v2(db));
    CHECK(cuMemFree_v2(dc));
    CHECK(cuCtxDestroy_v2(ctx));
    free(each);
    return 0;
}
