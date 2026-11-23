/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * vecadd_client.c - a plain CUDA Driver API program, written the way a
 * CUDA-10-era application is: it knows nothing about the adapter and links
 * only `libcuda`. Built for x86_64 (run under Rosetta) and arm64.
 *
 * usage: vecadd_client <vecadd cubin or PTX file> [kernel name]
 * Exercises: init, device queries, context, module load from memory, the
 * three launch forms (separate pointers, packed buffer, CUDA 3.x cuParamSet*),
 * interior device pointers, a transfer larger than one wire chunk, memset,
 * host memory, error paths. Exits 0 only if every check passes.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

CUresult cuInit(unsigned int);
CUresult cuDriverGetVersion(int *);
CUresult cuDeviceGetCount(int *);
CUresult cuDeviceGet(CUdevice *, int);
CUresult cuDeviceGetName(char *, int, CUdevice);
CUresult cuDeviceComputeCapability(int *, int *, CUdevice);
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
CUresult cuCtxDestroy_v2(CUcontext);
CUresult cuCtxSynchronize(void);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
CUresult cuMemsetD32_v2(CUdeviceptr, unsigned int, size_t);
CUresult cuMemAllocHost_v2(void **, size_t);
CUresult cuMemFreeHost(void *);
CUresult cuModuleLoadData(CUmodule *, const void *);
CUresult cuModuleUnload(CUmodule);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream,
                        void **, void **);
CUresult cuParamSetSize(CUfunction, unsigned int);
CUresult cuParamSeti(CUfunction, int, unsigned int);
CUresult cuParamSetv(CUfunction, int, void *, unsigned int);
CUresult cuFuncSetBlockShape(CUfunction, int, int, int);
CUresult cuLaunchGrid(CUfunction, int, int);
CUresult cuGetErrorName(CUresult, const char **);
CUresult cuGLInit(void); /* one of the generated "not supported" stubs */

static int g_failed;

#define CHECK(call)                                                                 \
    do {                                                                            \
        CUresult rc_ = (call);                                                      \
        if (rc_ != 0) {                                                             \
            const char *name_ = "?";                                                \
            cuGetErrorName(rc_, &name_);                                            \
            fprintf(stderr, "FAIL %s -> %d (%s)\n", #call, rc_, name_ ? name_ : "?"); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

#define EXPECT(cond)                                  \
    do {                                              \
        if (!(cond)) {                                \
            fprintf(stderr, "FAIL expected %s\n", #cond); \
            g_failed = 1;                             \
        }                                             \
    } while (0)

static int check_sum(const char *what, const float *c, const float *a, const float *b, unsigned n, unsigned total) {
    for (unsigned i = 0; i < total; i++) {
        float want = i < n ? a[i] + b[i] : 0.0f;
        if (c[i] != want) {
            fprintf(stderr, "FAIL %s: c[%u] = %g, expected %g\n", what, i, c[i], want);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vecadd cubin or PTX> [kernel]\n", argv[0]);
        return 2;
    }
    const char *kernel = argc > 2 ? argv[2] : "vecadd_macos";
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long image_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    /* One spare NUL so a PTX file is a C string, as cuModuleLoadData expects. */
    char *image = calloc(1, (size_t)image_len + 1);
    if (!image || fread(image, 1, (size_t)image_len, f) != (size_t)image_len) return 2;
    fclose(f);

    enum { N = 1024, TOTAL = 1280 };
    static float a[TOTAL], b[TOTAL], c[TOTAL];
    for (unsigned i = 0; i < TOTAL; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = 1000.0f - (float)i;
    }

    int version = 0, count = 0, major = 0, minor = 0;
    char name[128];
    CUdevice dev;
    CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDriverGetVersion(&version));
    CHECK(cuDeviceGetCount(&count));
    EXPECT(count >= 1);
    EXPECT(cuDeviceGet(&dev, count) == 101); /* CUDA_ERROR_INVALID_DEVICE */
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    CHECK(cuDeviceComputeCapability(&major, &minor, dev));
    printf("device: %s (sm_%d%d), driver %d, %zu-bit client\n", name, major, minor, version, sizeof(void *) * 8);
    CHECK(cuCtxCreate_v2(&ctx, 0, dev));

    CUmodule module;
    CUfunction fn;
    CHECK(cuModuleLoadData(&module, image));
    EXPECT(cuModuleGetFunction(&fn, module, "no_such_kernel") == 500); /* CUDA_ERROR_NOT_FOUND */
    CHECK(cuModuleGetFunction(&fn, module, kernel));

    CUdeviceptr da, db, dc;
    CHECK(cuMemAlloc_v2(&da, sizeof(a)));
    CHECK(cuMemAlloc_v2(&db, sizeof(b)));
    CHECK(cuMemAlloc_v2(&dc, sizeof(c)));
    CHECK(cuMemcpyHtoD_v2(da, a, sizeof(a)));
    CHECK(cuMemcpyHtoD_v2(db, b, sizeof(b)));
    unsigned n = N;

    /* 1. Separate parameter pointers (the usual form). */
    CHECK(cuMemsetD32_v2(dc, 0, TOTAL));
    void *params[] = {&da, &db, &dc, &n};
    CHECK(cuLaunchKernel(fn, (TOTAL + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, params, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH_v2(c, dc, sizeof(c)));
    g_failed |= check_sum("separate params", c, a, b, N, TOTAL);

    /* 2. Packed parameter buffer through `extra`. */
    struct {
        CUdeviceptr a, b, c;
        unsigned n;
    } packed = {da, db, dc, 512};
    size_t packed_size = 28;
    void *extra[] = {(void *)1, &packed, (void *)2, &packed_size, (void *)0};
    CHECK(cuMemsetD32_v2(dc, 0, TOTAL));
    CHECK(cuLaunchKernel(fn, (TOTAL + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, NULL, extra));
    CHECK(cuMemcpyDtoH_v2(c, dc, sizeof(c)));
    g_failed |= check_sum("packed params", c, a, b, 512, TOTAL);

    /* 3. CUDA 3.x protocol: cuParamSet* at explicit offsets, then cuLaunchGrid. */
    CHECK(cuMemsetD32_v2(dc, 0, TOTAL));
    CHECK(cuParamSetv(fn, 0, &da, 8));
    CHECK(cuParamSetv(fn, 8, &db, 8));
    CHECK(cuParamSetv(fn, 16, &dc, 8));
    CHECK(cuParamSeti(fn, 24, 300));
    CHECK(cuParamSetSize(fn, 28));
    CHECK(cuFuncSetBlockShape(fn, 256, 1, 1));
    CHECK(cuLaunchGrid(fn, (TOTAL + 255) / 256, 1));
    CHECK(cuMemcpyDtoH_v2(c, dc, sizeof(c)));
    g_failed |= check_sum("cuParamSet protocol", c, a, b, 300, TOTAL);

    /* 4. Interior device pointers: write and read back the middle of a buffer. */
    float patch[4] = {-1.0f, -2.0f, -3.0f, -4.0f}, back[4] = {0};
    CHECK(cuMemcpyHtoD_v2(dc + 100 * sizeof(float), patch, sizeof(patch)));
    CHECK(cuMemcpyDtoH_v2(back, dc + 100 * sizeof(float), sizeof(back)));
    EXPECT(memcmp(patch, back, sizeof(patch)) == 0);
    EXPECT(cuMemcpyHtoD_v2(dc + sizeof(c) - 4, patch, sizeof(patch)) != 0); /* runs past the allocation */

    /* 5. A round trip from page-locked host memory. The default 40 MiB spans
     * several 16 MiB wire chunks; VECADD_BIG_MIB sets the size (0 skips it),
     * for a first run on a card whose staging path is slow. */
    const char *big_env = getenv("VECADD_BIG_MIB");
    size_t big = (size_t)(big_env ? strtoul(big_env, NULL, 10) : 40) * 1024u * 1024u;
    void *host_out = NULL, *host_in = NULL;
    CUdeviceptr dbig;
    if (big == 0) goto skip_big;
    CHECK(cuMemAllocHost_v2(&host_out, big));
    CHECK(cuMemAllocHost_v2(&host_in, big));
    for (size_t i = 0; i < big / 4; i++) ((uint32_t *)host_out)[i] = (uint32_t)(i * 2654435761u);
    CHECK(cuMemAlloc_v2(&dbig, big));
    CHECK(cuMemcpyHtoD_v2(dbig, host_out, big));
    CHECK(cuMemcpyDtoH_v2(host_in, dbig, big));
    EXPECT(memcmp(host_out, host_in, big) == 0);
    CHECK(cuMemFree_v2(dbig));
    CHECK(cuMemFreeHost(host_out));
    CHECK(cuMemFreeHost(host_in));
skip_big:

    /* 6. Error paths. */
    EXPECT(cuMemFree_v2(da + 8) != 0);  /* not an allocation base */
    EXPECT(cuGLInit() == 801);          /* CUDA_ERROR_NOT_SUPPORTED */
    void *wrong[] = {&da, &db, &dc, &n};
    EXPECT(cuLaunchKernel(fn, 0, 1, 1, 256, 1, 1, 0, NULL, wrong, NULL) != 0); /* zero grid */

    CHECK(cuMemFree_v2(da));
    CHECK(cuMemFree_v2(db));
    CHECK(cuMemFree_v2(dc));
    CHECK(cuModuleUnload(module));
    CHECK(cuCtxDestroy_v2(ctx));
    free(image);
    if (g_failed) {
        fprintf(stderr, "vecadd_client: FAILED\n");
        return 1;
    }
    printf("vecadd_client: PASS\n");
    return 0;
}
