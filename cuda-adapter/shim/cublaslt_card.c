/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cublaslt_card.c - cublasLtMatmul that runs ON THE CARD.
 *
 * The cuBLASLt entry points MLX's CUDA backend calls, with the matrix product
 * done by launching the repo's `matmul_macos` kernel (sm_120) through the CUDA
 * Driver API. A, B and D are real device pointers, as with NVIDIA's library;
 * nothing here reads or writes them on the host. (portkit/cuda-mathlib-stubs
 * has the CPU variant, which only works when "device" pointers are host
 * memory.)
 *
 * What is computed: D = A * B for single precision, column-major, packed
 * matrices (leading dimension == rows), no transposes, one batch, alpha == 1,
 * beta == 0, default epilogue. Every other configuration returns
 * CUBLAS_STATUS_NOT_SUPPORTED; nothing is approximated or silently changed.
 *
 * How column-major maps onto the kernel: `matmul_macos` computes
 * C[M,N] = A[M,K] @ B[K,N] in row-major storage. A packed column-major m-by-k
 * matrix is byte for byte the row-major k-by-m transpose, so
 *     D (m x n) = A (m x k) * B (k x n)            column-major
 * is the same memory as
 *     D^T (n x m) = B^T (n x k) @ A^T (k x m)      row-major,
 * i.e. kernel(first = B, second = A, out = D, M = n, N = m, K = k).
 *
 * The kernel image is found at $CUDA_ADAPTER_MATMUL_CUBIN, else at
 * kernels/matmul_macos.cubin next to this library.
 */
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LT_API __attribute__((visibility("default")))

/* ---- the cuBLASLt types this file needs (ABI as in cublasLt.h) ---------- */

typedef int cublasStatus_t;
enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
};
enum { CUDA_R_32F = 0 };
enum { CUBLAS_COMPUTE_32F = 68 };
enum { CUBLAS_OP_N = 0 };
enum { CUBLASLT_EPILOGUE_DEFAULT = 1 };
enum {
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 23,
};
enum {
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 4,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 5,
};
enum { CUBLASLT_ORDER_COL = 0 };

typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

#define MAGIC_HANDLE 0x4c744831u /* "LtH1" */
#define MAGIC_DESC 0x4c744431u
#define MAGIC_LAYOUT 0x4c744c31u
#define MAGIC_PREF 0x4c745031u

typedef struct {
    uint32_t magic;
} lt_handle;

typedef struct {
    uint32_t magic;
    int compute_type;
    int scale_type;
    int pointer_mode;
    int trans_a;
    int trans_b;
    int epilogue;
    int unsupported_attribute; /* an attribute this file cannot honour was set */
} lt_desc;

typedef struct {
    uint32_t magic;
    int data_type;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
    int order;
    int batch_count;
} lt_layout;

typedef struct {
    uint32_t magic;
} lt_pref;

/* ---- the CUDA Driver API (the adapter's stand-in libcuda) --------------- */

typedef unsigned long long CUdeviceptr;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
int cuModuleLoad(CUmodule *, const char *);
int cuModuleGetFunction(CUfunction *, CUmodule, const char *);
int cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                   void **);

/* ---- kernel module, loaded once ----------------------------------------- */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static CUfunction g_matmul;
static int g_load_failed;

static void own_directory(char *out, size_t cap) {
    Dl_info info;
    out[0] = 0;
    if (dladdr((void *)&own_directory, &info) && info.dli_fname) {
        snprintf(out, cap, "%s", info.dli_fname);
        char *slash = strrchr(out, '/');
        if (slash) *slash = 0;
        else out[0] = 0;
    }
}

/* Returns the kernel, loading its module on first use, or NULL. */
static CUfunction matmul_kernel(void) {
    pthread_mutex_lock(&g_lock);
    if (!g_matmul && !g_load_failed) {
        char path[1024];
        const char *env = getenv("CUDA_ADAPTER_MATMUL_CUBIN");
        if (env && *env) {
            snprintf(path, sizeof(path), "%s", env);
        } else {
            char dir[900];
            own_directory(dir, sizeof(dir));
            snprintf(path, sizeof(path), "%s/kernels/matmul_macos.cubin", dir);
        }
        CUmodule module = NULL;
        int rc = cuModuleLoad(&module, path);
        if (rc == 0) rc = cuModuleGetFunction(&g_matmul, module, "matmul_macos");
        if (rc != 0) {
            g_load_failed = 1;
            g_matmul = NULL;
            fprintf(stderr, "cublasLt (card): cannot load the matmul kernel from %s (CUDA error %d)\n", path, rc);
        }
    }
    CUfunction fn = g_matmul;
    pthread_mutex_unlock(&g_lock);
    return fn;
}

/* ---- handles ------------------------------------------------------------ */

LT_API cublasStatus_t cublasLtCreate(void **handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    lt_handle *h = (lt_handle *)calloc(1, sizeof(*h));
    if (!h) return CUBLAS_STATUS_ALLOC_FAILED;
    h->magic = MAGIC_HANDLE;
    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtDestroy(void *handle) {
    lt_handle *h = (lt_handle *)handle;
    if (!h || h->magic != MAGIC_HANDLE) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->magic = 0;
    free(h);
    return CUBLAS_STATUS_SUCCESS;
}

LT_API size_t cublasLtGetVersion(void) { return 120400; }

LT_API size_t cublasLtGetCudartVersion(void) { return 12040; }

LT_API const char *cublasLtGetStatusString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

LT_API cublasStatus_t cublasLtMatmulDescCreate(void **desc, int compute_type, int scale_type) {
    if (!desc) return CUBLAS_STATUS_INVALID_VALUE;
    lt_desc *d = (lt_desc *)calloc(1, sizeof(*d));
    if (!d) return CUBLAS_STATUS_ALLOC_FAILED;
    d->magic = MAGIC_DESC;
    d->compute_type = compute_type;
    d->scale_type = scale_type;
    d->trans_a = CUBLAS_OP_N;
    d->trans_b = CUBLAS_OP_N;
    d->epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    *desc = d;
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtMatmulDescDestroy(void *desc) {
    lt_desc *d = (lt_desc *)desc;
    if (!d || d->magic != MAGIC_DESC) return CUBLAS_STATUS_INVALID_VALUE;
    d->magic = 0;
    free(d);
    return CUBLAS_STATUS_SUCCESS;
}

static int read_int(const void *buf, size_t size, int *out) {
    if (!buf || size != sizeof(int)) return 0;
    memcpy(out, buf, sizeof(int));
    return 1;
}

/* Attributes are recorded here and judged in cublasLtMatmul, as the real
 * library does. One this file does not model marks the descriptor so the
 * product is refused instead of being computed without it. */
LT_API cublasStatus_t cublasLtMatmulDescSetAttribute(void *desc, int attr, const void *buf, size_t size) {
    lt_desc *d = (lt_desc *)desc;
    if (!d || d->magic != MAGIC_DESC || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    int value = 0;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_POINTER_MODE:
        return read_int(buf, size, &d->pointer_mode) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    case CUBLASLT_MATMUL_DESC_TRANSA:
        return read_int(buf, size, &d->trans_a) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    case CUBLASLT_MATMUL_DESC_TRANSB:
        return read_int(buf, size, &d->trans_b) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    case CUBLASLT_MATMUL_DESC_EPILOGUE:
        return read_int(buf, size, &d->epilogue) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    default:
        (void)value;
        d->unsupported_attribute = 1;
        return CUBLAS_STATUS_SUCCESS;
    }
}

LT_API cublasStatus_t cublasLtMatrixLayoutCreate(void **layout, int data_type, uint64_t rows, uint64_t cols,
                                                 int64_t ld) {
    if (!layout) return CUBLAS_STATUS_INVALID_VALUE;
    lt_layout *l = (lt_layout *)calloc(1, sizeof(*l));
    if (!l) return CUBLAS_STATUS_ALLOC_FAILED;
    l->magic = MAGIC_LAYOUT;
    l->data_type = data_type;
    l->rows = rows;
    l->cols = cols;
    l->ld = ld;
    l->order = CUBLASLT_ORDER_COL;
    l->batch_count = 1;
    *layout = l;
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtMatrixLayoutDestroy(void *layout) {
    lt_layout *l = (lt_layout *)layout;
    if (!l || l->magic != MAGIC_LAYOUT) return CUBLAS_STATUS_INVALID_VALUE;
    l->magic = 0;
    free(l);
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtMatrixLayoutSetAttribute(void *layout, int attr, const void *buf, size_t size) {
    lt_layout *l = (lt_layout *)layout;
    if (!l || l->magic != MAGIC_LAYOUT || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_ORDER:
        return read_int(buf, size, &l->order) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
        return read_int(buf, size, &l->batch_count) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
        return CUBLAS_STATUS_SUCCESS; /* only meaningful with batch_count > 1, which is refused */
    default:
        l->batch_count = -1; /* unknown layout attribute: refuse the product */
        return CUBLAS_STATUS_SUCCESS;
    }
}

LT_API cublasStatus_t cublasLtMatmulPreferenceCreate(void **pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    lt_pref *p = (lt_pref *)calloc(1, sizeof(*p));
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    p->magic = MAGIC_PREF;
    *pref = p;
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtMatmulPreferenceDestroy(void *pref) {
    lt_pref *p = (lt_pref *)pref;
    if (!p || p->magic != MAGIC_PREF) return CUBLAS_STATUS_INVALID_VALUE;
    p->magic = 0;
    free(p);
    return CUBLAS_STATUS_SUCCESS;
}

/* The kernel needs no workspace, so the preference has nothing to record. */
LT_API cublasStatus_t cublasLtMatmulPreferenceSetAttribute(void *pref, int attr, const void *buf, size_t size) {
    lt_pref *p = (lt_pref *)pref;
    (void)attr;
    (void)size;
    if (!p || p->magic != MAGIC_PREF || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    return CUBLAS_STATUS_SUCCESS;
}

/* ---- the product -------------------------------------------------------- */

typedef struct {
    uint32_t m, n, k;
} lt_shape;

/* Decide whether this file can compute the described product. */
static cublasStatus_t admit(const lt_desc *d, const lt_layout *a, const lt_layout *b, const lt_layout *c,
                            const lt_layout *out, lt_shape *shape) {
    if (!d || d->magic != MAGIC_DESC || !a || a->magic != MAGIC_LAYOUT || !b || b->magic != MAGIC_LAYOUT || !out ||
        out->magic != MAGIC_LAYOUT || (c && c->magic != MAGIC_LAYOUT))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (d->compute_type != CUBLAS_COMPUTE_32F || d->scale_type != CUDA_R_32F || d->unsupported_attribute ||
        d->pointer_mode != 0 || d->trans_a != CUBLAS_OP_N || d->trans_b != CUBLAS_OP_N ||
        d->epilogue != CUBLASLT_EPILOGUE_DEFAULT)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    const lt_layout *all[3] = {a, b, out};
    for (int i = 0; i < 3; i++) {
        const lt_layout *l = all[i];
        if (l->data_type != CUDA_R_32F || l->order != CUBLASLT_ORDER_COL || l->batch_count != 1 || l->rows == 0 ||
            l->cols == 0 || l->ld != (int64_t)l->rows)
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (a->cols != b->rows || out->rows != a->rows || out->cols != b->cols) return CUBLAS_STATUS_INVALID_VALUE;
    /* The kernel takes 32-bit sizes and its grid.y (rows of the row-major
     * output, here n) is limited to 65535 blocks of 16. */
    if (a->rows > 0x7fffffffu || a->cols > 0x7fffffffu || b->cols > 0x7fffffffu || (b->cols + 15) / 16 > 65535)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    shape->m = (uint32_t)a->rows;
    shape->k = (uint32_t)a->cols;
    shape->n = (uint32_t)b->cols;
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtMatmulAlgoGetHeuristic(void *handle, void *desc, void *a, void *b, void *c, void *d,
                                                     void *pref, int requested, cublasLtMatmulHeuristicResult_t *results,
                                                     int *returned) {
    lt_handle *h = (lt_handle *)handle;
    (void)pref;
    if (!h || h->magic != MAGIC_HANDLE) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!results || !returned || requested < 1) return CUBLAS_STATUS_INVALID_VALUE;
    *returned = 0;
    lt_shape shape;
    cublasStatus_t rc = admit((lt_desc *)desc, (lt_layout *)a, (lt_layout *)b, (lt_layout *)c, (lt_layout *)d, &shape);
    if (rc != CUBLAS_STATUS_SUCCESS) return rc;
    memset(&results[0], 0, sizeof(results[0]));
    results[0].algo.data[0] = 1; /* the one algorithm: the matmul_macos kernel */
    results[0].workspaceSize = 0;
    results[0].state = CUBLAS_STATUS_SUCCESS;
    results[0].wavesCount = 1.0f;
    *returned = 1;
    return CUBLAS_STATUS_SUCCESS;
}

LT_API cublasStatus_t cublasLtMatmul(void *handle, void *desc, const void *alpha, const void *A, void *a_desc,
                                     const void *B, void *b_desc, const void *beta, const void *C, void *c_desc,
                                     void *D, void *d_desc, const cublasLtMatmulAlgo_t *algo, void *workspace,
                                     size_t workspace_bytes, void *stream) {
    lt_handle *h = (lt_handle *)handle;
    (void)algo;
    (void)workspace;
    (void)workspace_bytes;
    (void)stream; /* the served stack completes each launch before it returns */
    if (!h || h->magic != MAGIC_HANDLE) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!A || !B || !D) return CUBLAS_STATUS_INVALID_VALUE;
    lt_shape shape;
    cublasStatus_t rc =
        admit((lt_desc *)desc, (lt_layout *)a_desc, (lt_layout *)b_desc, (lt_layout *)c_desc, (lt_layout *)d_desc, &shape);
    if (rc != CUBLAS_STATUS_SUCCESS) return rc;
    /* alpha and beta are host scalars (pointer mode HOST was checked). Only
     * the plain product is computed: C is not read. */
    float alpha_value = 1.0f, beta_value = 0.0f;
    if (alpha) memcpy(&alpha_value, alpha, sizeof(float));
    if (beta) memcpy(&beta_value, beta, sizeof(float));
    if (alpha_value != 1.0f || beta_value != 0.0f) return CUBLAS_STATUS_NOT_SUPPORTED;
    (void)C;
    /* The kernel reads its inputs while it writes the output. */
    if (D == A || D == B) return CUBLAS_STATUS_NOT_SUPPORTED;

    CUfunction kernel = matmul_kernel();
    if (!kernel) return CUBLAS_STATUS_INTERNAL_ERROR;

    /* Column-major D = A * B is row-major D^T = B^T @ A^T (see the file header). */
    CUdeviceptr first = (CUdeviceptr)(uintptr_t)B, second = (CUdeviceptr)(uintptr_t)A, out = (CUdeviceptr)(uintptr_t)D;
    uint32_t rows = shape.n, cols = shape.m, inner = shape.k;
    void *params[] = {&first, &second, &out, &rows, &cols, &inner};
    int launched = cuLaunchKernel(kernel, (cols + 15) / 16, (rows + 15) / 16, 1, 16, 16, 1, 0, NULL, params, NULL);
    return launched == 0 ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}
