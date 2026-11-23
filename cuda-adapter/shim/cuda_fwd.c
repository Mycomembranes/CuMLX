/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cuda_fwd.c - stand-in libcuda: forwards the CUDA Driver API to
 * cuda_adapter_server over a private AF_UNIX socket (cuda-oxide.driver.v1).
 *
 * One source, two slices:
 *   x86_64  for CUDA-10-era applications running under Rosetta. Un-suffixed
 *           names (cuMemAlloc, cuMemcpyHtoD, ...) keep the CUDA 3.x ABI that
 *           the real 10.1 library exports for them: 32-bit CUdeviceptr and
 *           `unsigned int` sizes. Applications built against the 10.1 cuda.h
 *           call the _v2 names through macros and never see this.
 *   arm64   for native clients. No legacy arm64 macOS CUDA ever existed, so
 *           un-suffixed names are the 64-bit ABI, matching libcuda_driverkit.
 *
 * Every exported function is marked FWD_API. scripts/gen_unsupported.py reads
 * those markers and generates a stub returning CUDA_ERROR_NOT_SUPPORTED for
 * each remaining symbol of the CUDA 10.1 export list, so the library exports
 * the complete list.
 *
 * Environment:
 *   CUDA_ADAPTER_SOCKET             server socket (default /tmp/cuda-adapter-<uid>.sock)
 *   CUDA_ADAPTER_TRACE=1            log every cu* call to stderr (or CUDA_ADAPTER_TRACE_FILE)
 *   CUDA_ADAPTER_TRACE_PERMISSIVE=1 unsupported calls return CUDA_SUCCESS, for
 *                                   sizing an application's API surface only:
 *                                   their outputs are NOT filled in.
 */

#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include "fwd_wire.h"

#define FWD_API __attribute__((visibility("default")))

#if defined(__x86_64__)
#define FWD_LEGACY_V1 1
#else
#define FWD_LEGACY_V1 0
#endif

/* ---- CUDA types and codes (CUDA 10.1 cuda.h) ---------------------------- */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

#if FWD_LEGACY_V1
typedef unsigned int fwd_dptr1; /* CUDA 3.x CUdeviceptr */
typedef unsigned int fwd_size1; /* CUDA 3.x byte counts */
#else
typedef CUdeviceptr fwd_dptr1;
typedef size_t fwd_size1;
#endif

enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999,
};

/* CUjit_option values used by cuModuleLoadDataEx. */
enum {
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
};

/* cuLaunchKernel `extra` protocol. */
#define CU_LAUNCH_PARAM_END ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)

#define FWD_CLIENT_ABI 1
#define FWD_DEFAULT_CHUNK (16u * 1024u * 1024u)
#define FWD_MAX_PARAMS 64
#define FWD_LEGACY_PARAM_BYTES 4096

/* ---- tracing ------------------------------------------------------------ */

static int g_trace = -1; /* -1 unknown, 0 off, 1 on */
static int g_permissive;
static FILE *g_trace_out;

static void trace_setup(void) {
    const char *on = getenv("CUDA_ADAPTER_TRACE");
    const char *permissive = getenv("CUDA_ADAPTER_TRACE_PERMISSIVE");
    const char *file = getenv("CUDA_ADAPTER_TRACE_FILE");
    g_permissive = permissive && *permissive && strcmp(permissive, "0") != 0;
    g_trace = (on && *on && strcmp(on, "0") != 0) || g_permissive;
    g_trace_out = stderr;
    if (g_trace && file && *file) {
        FILE *f = fopen(file, "a");
        if (f) {
            setvbuf(f, NULL, _IOLBF, 0);
            g_trace_out = f;
        }
    }
}

static void trace_call(const char *name) {
    if (g_trace < 0) trace_setup();
    if (g_trace) fprintf(g_trace_out, "cuda-adapter: call %s\n", name);
}

#define TRACE() trace_call(__func__)

static void fwd_note(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("cuda-adapter: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* Called by every generated stub (see generated/cuda_unsupported.c). */
__attribute__((visibility("hidden"))) CUresult fwd_unsupported(const char *name, int *warned) {
    if (g_trace < 0) trace_setup();
    if (g_trace) fprintf(g_trace_out, "cuda-adapter: call %s (unsupported)\n", name);
    if (!*warned) {
        *warned = 1;
        if (!g_trace) fwd_note("%s is not supported by the adapter", name);
    }
    return g_permissive ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}

/* ---- connection and RPC ------------------------------------------------- */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_fd = -1;
static int g_connect_failed; /* reported once */
static uint32_t g_seq;
static size_t g_chunk = FWD_DEFAULT_CHUNK;
static int g_shared_host; /* server offers shared host segments */

typedef struct {
    uint8_t *body; /* malloc'd reply frame; owner frees via reply_free */
    mp_cur root;
    mp_cur result;
    int has_result;
    char error[512];
} fwd_reply;

static void reply_free(fwd_reply *r) {
    free(r->body);
    r->body = NULL;
}

static void req_begin(mp_buf *b, const char *op, uint32_t nargs) {
    memset(b, 0, sizeof(*b));
    mp_map(b, 3);
    mp_str(b, "op");
    mp_str(b, op);
    mp_str(b, "seq");
    mp_uint(b, ++g_seq);
    mp_str(b, "args");
    mp_map(b, nargs);
}

static void arg_uint(mp_buf *b, const char *key, uint64_t v) {
    mp_str(b, key);
    mp_uint(b, v);
}

/* Send one request and read its reply. g_lock must be held. */
static CUresult exchange_locked(mp_buf *req, fwd_reply *reply) {
    memset(reply, 0, sizeof(*reply));
    if (req->failed) {
        free(req->data);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    int sent = fwd_send_frame(g_fd, req);
    free(req->data);
    size_t len = 0;
    if (sent != 0 || !(reply->body = fwd_recv_frame(g_fd, &len))) {
        fwd_note("lost the connection to cuda_adapter_server");
        close(g_fd);
        g_fd = -1;
        return CUDA_ERROR_UNKNOWN;
    }
    reply->root.p = reply->body;
    reply->root.end = reply->body + len;
    mp_cur cur;
    CUresult code = CUDA_ERROR_UNKNOWN;
    if (mp_find(reply->root, "cu", &cur)) {
        code = (CUresult)mp_read_int(&cur);
        if (cur.bad) code = CUDA_ERROR_UNKNOWN;
    }
    if (mp_find(reply->root, "error", &cur)) {
        size_t n;
        const uint8_t *text = mp_read_bytes(&cur, &n, 1);
        if (text) {
            if (n >= sizeof(reply->error)) n = sizeof(reply->error) - 1;
            memcpy(reply->error, text, n);
        }
    }
    reply->has_result = mp_find(reply->root, "result", &reply->result);
    return code;
}

static const char *socket_path(char *buf, size_t n) {
    const char *env = getenv("CUDA_ADAPTER_SOCKET");
    if (env && *env) return env;
    snprintf(buf, n, "/tmp/cuda-adapter-%u.sock", (unsigned)geteuid());
    return buf;
}

/* Connect and greet. g_lock must be held. */
static CUresult connect_locked(void) {
    if (g_fd >= 0) return CUDA_SUCCESS;
    char buf[128];
    const char *path = socket_path(buf, sizeof(buf));
    g_fd = fwd_connect_unix(path);
    if (g_fd < 0) {
        if (!g_connect_failed) {
            g_connect_failed = 1;
            fwd_note("no cuda_adapter_server at %s (%s); start one or set CUDA_ADAPTER_SOCKET", path,
                     strerror(errno));
        }
        return CUDA_ERROR_NO_DEVICE;
    }
    mp_buf req;
    req_begin(&req, "hello", 2);
    arg_uint(&req, "abi", FWD_CLIENT_ABI);
    mp_str(&req, "arch");
#if defined(__x86_64__)
    mp_str(&req, "x86_64");
#else
    mp_str(&req, "arm64");
#endif
    fwd_reply reply;
    CUresult rc = exchange_locked(&req, &reply);
    if (rc == CUDA_SUCCESS && reply.has_result) {
        mp_cur cur;
        if (mp_find(reply.result, "max_chunk", &cur)) {
            uint64_t chunk = (uint64_t)mp_read_int(&cur);
            if (!cur.bad && chunk >= 4096 && chunk <= FWD_DEFAULT_CHUNK) g_chunk = (size_t)chunk;
        }
        const char *plain = getenv("CUDA_ADAPTER_NO_SHARED_HOST");
        g_shared_host = mp_find(reply.result, "shared_host_memory", &cur) && !cur.bad && cur.p < cur.end &&
                        *cur.p == 0xc3 && !(plain && *plain && strcmp(plain, "0") != 0);
    } else if (rc != CUDA_SUCCESS) {
        fwd_note("cuda_adapter_server refused the connection: %s", reply.error[0] ? reply.error : "no reason given");
        if (g_fd >= 0) close(g_fd);
        g_fd = -1;
        rc = CUDA_ERROR_NO_DEVICE;
    }
    reply_free(&reply);
    return rc;
}

/* Full round trip. The caller frees the reply with reply_free. */
static CUresult rpc(mp_buf *req, fwd_reply *reply) {
    pthread_mutex_lock(&g_lock);
    CUresult rc = connect_locked();
    if (rc == CUDA_SUCCESS) {
        rc = exchange_locked(req, reply);
    } else {
        free(req->data);
        memset(reply, 0, sizeof(*reply));
    }
    pthread_mutex_unlock(&g_lock);
    return rc;
}

/* Round trip whose reply carries nothing the caller needs. */
static CUresult rpc_void(mp_buf *req) {
    fwd_reply reply;
    CUresult rc = rpc(req, &reply);
    reply_free(&reply);
    return rc;
}

/* Round trip returning one unsigned integer result. */
static CUresult rpc_u64(mp_buf *req, uint64_t *out) {
    fwd_reply reply;
    CUresult rc = rpc(req, &reply);
    if (rc == CUDA_SUCCESS) {
        *out = (uint64_t)mp_read_int(&reply.result);
        if (!reply.has_result || reply.result.bad) rc = CUDA_ERROR_UNKNOWN;
    }
    reply_free(&reply);
    return rc;
}

/* ---- local state -------------------------------------------------------- */

typedef struct fwd_function {
    uint64_t remote;
    uint32_t nparams;
    uint32_t sizes[FWD_MAX_PARAMS];
    uint32_t offsets[FWD_MAX_PARAMS];
    /* CUDA 3.x launch state (cuParamSet*, cuFuncSetBlockShape, cuLaunchGrid). */
    uint8_t legacy[FWD_LEGACY_PARAM_BYTES];
    uint32_t legacy_size;
    uint32_t block[3];
    uint32_t shared;
} fwd_function;

typedef struct {
    struct timespec at;
    int recorded;
} fwd_event;

static CUcontext g_ctx_stack[16];
static int g_ctx_depth;
static CUdevice g_ctx_device;
static CUcontext g_primary_ctx;
static int g_primary_refs;

static CUcontext current_ctx(void) { return g_ctx_depth ? g_ctx_stack[g_ctx_depth - 1] : NULL; }

/* ---- initialisation and devices ---------------------------------------- */

FWD_API CUresult cuInit(unsigned int flags) {
    TRACE();
    mp_buf req;
    req_begin(&req, "init", 1);
    arg_uint(&req, "flags", flags);
    return rpc_void(&req);
}

FWD_API CUresult cuDriverGetVersion(int *version) {
    TRACE();
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    mp_buf req;
    req_begin(&req, "driver_version", 0);
    uint64_t v = 0;
    CUresult rc = rpc_u64(&req, &v);
    if (rc == CUDA_SUCCESS) *version = (int)v;
    return rc;
}

FWD_API CUresult cuDeviceGetCount(int *count) {
    TRACE();
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    mp_buf req;
    req_begin(&req, "device_count", 0);
    uint64_t n = 0;
    CUresult rc = rpc_u64(&req, &n);
    if (rc == CUDA_SUCCESS) *count = (int)n;
    return rc;
}

FWD_API CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    TRACE();
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    int count = 0;
    CUresult rc = cuDeviceGetCount(&count);
    if (rc != CUDA_SUCCESS) return rc;
    if (ordinal < 0 || ordinal >= count) return CUDA_ERROR_INVALID_DEVICE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

typedef struct {
    char name[256];
    uint64_t total_mem;
    int has_total;
    int sm_major;
    int sm_minor;
} fwd_device_info;

static CUresult device_info(CUdevice dev, fwd_device_info *info) {
    memset(info, 0, sizeof(*info));
    if (dev < 0) return CUDA_ERROR_INVALID_DEVICE;
    mp_buf req;
    req_begin(&req, "device_info", 1);
    arg_uint(&req, "device", (uint64_t)dev);
    fwd_reply reply;
    CUresult rc = rpc(&req, &reply);
    if (rc == CUDA_SUCCESS) {
        mp_cur cur;
        size_t n;
        if (mp_find(reply.result, "name", &cur)) {
            const uint8_t *text = mp_read_bytes(&cur, &n, 1);
            if (text) {
                if (n >= sizeof(info->name)) n = sizeof(info->name) - 1;
                memcpy(info->name, text, n);
            }
        }
        if (mp_find(reply.result, "total_mem", &cur) && !mp_is_nil(cur)) {
            info->total_mem = (uint64_t)mp_read_int(&cur);
            info->has_total = !cur.bad;
        }
        if (mp_find(reply.result, "sm_major", &cur)) info->sm_major = (int)mp_read_int(&cur);
        if (mp_find(reply.result, "sm_minor", &cur)) info->sm_minor = (int)mp_read_int(&cur);
    }
    reply_free(&reply);
    return rc;
}

FWD_API CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    TRACE();
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    fwd_device_info info;
    CUresult rc = device_info(dev, &info);
    if (rc == CUDA_SUCCESS) snprintf(name, (size_t)len, "%s", info.name);
    return rc;
}

FWD_API CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    TRACE();
    if (!major || !minor) return CUDA_ERROR_INVALID_VALUE;
    fwd_device_info info;
    CUresult rc = device_info(dev, &info);
    if (rc == CUDA_SUCCESS) {
        *major = info.sm_major;
        *minor = info.sm_minor;
    }
    return rc;
}

FWD_API CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    TRACE();
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    fwd_device_info info;
    CUresult rc = device_info(dev, &info);
    if (rc != CUDA_SUCCESS) return rc;
    if (!info.has_total) return CUDA_ERROR_NOT_SUPPORTED;
    *bytes = (size_t)info.total_mem;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuDeviceTotalMem(fwd_size1 *bytes, CUdevice dev) {
    TRACE();
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    size_t total = 0;
    CUresult rc = cuDeviceTotalMem_v2(&total, dev);
    if (rc != CUDA_SUCCESS) return rc;
#if FWD_LEGACY_V1
    /* The 3.x ABI cannot express 4 GiB or more; report the largest value it can. */
    *bytes = total > 0xffffffffull ? 0xffffffffu : (unsigned int)total;
#else
    *bytes = total;
#endif
    return CUDA_SUCCESS;
}

FWD_API CUresult cuDeviceGetAttribute(int *value, int attribute, CUdevice dev) {
    TRACE();
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0) return CUDA_ERROR_INVALID_DEVICE;
    mp_buf req;
    req_begin(&req, "device_attr", 2);
    arg_uint(&req, "device", (uint64_t)dev);
    mp_str(&req, "attr");
    mp_int(&req, attribute);
    fwd_reply reply;
    CUresult rc = rpc(&req, &reply);
    if (rc == CUDA_SUCCESS) {
        *value = (int)mp_read_int(&reply.result);
        if (!reply.has_result || reply.result.bad) rc = CUDA_ERROR_UNKNOWN;
    }
    reply_free(&reply);
    return rc;
}

/* ---- contexts ----------------------------------------------------------- */

static CUresult ctx_create(CUcontext *ctx, unsigned int flags, CUdevice dev) {
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0) return CUDA_ERROR_INVALID_DEVICE;
    mp_buf req;
    req_begin(&req, "ctx_create", 2);
    arg_uint(&req, "device", (uint64_t)dev);
    arg_uint(&req, "flags", flags);
    uint64_t handle = 0;
    CUresult rc = rpc_u64(&req, &handle);
    if (rc != CUDA_SUCCESS) return rc;
    *ctx = (CUcontext)(uintptr_t)handle;
    g_ctx_device = dev;
    return CUDA_SUCCESS;
}

static void ctx_forget(CUcontext ctx) {
    int kept = 0;
    for (int i = 0; i < g_ctx_depth; i++)
        if (g_ctx_stack[i] != ctx) g_ctx_stack[kept++] = g_ctx_stack[i];
    g_ctx_depth = kept;
}

static CUresult ctx_destroy(CUcontext ctx) {
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    mp_buf req;
    req_begin(&req, "ctx_destroy", 1);
    arg_uint(&req, "ctx", (uint64_t)(uintptr_t)ctx);
    CUresult rc = rpc_void(&req);
    if (rc == CUDA_SUCCESS) ctx_forget(ctx);
    return rc;
}

static CUresult ctx_push(CUcontext ctx) {
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    if (g_ctx_depth >= (int)(sizeof(g_ctx_stack) / sizeof(g_ctx_stack[0]))) return CUDA_ERROR_INVALID_VALUE;
    g_ctx_stack[g_ctx_depth++] = ctx;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev) {
    TRACE();
    CUresult rc = ctx_create(ctx, flags, dev);
    return rc == CUDA_SUCCESS ? ctx_push(*ctx) : rc;
}

FWD_API CUresult cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice dev) {
    TRACE();
    CUresult rc = ctx_create(ctx, flags, dev);
    return rc == CUDA_SUCCESS ? ctx_push(*ctx) : rc;
}

FWD_API CUresult cuCtxDestroy_v2(CUcontext ctx) {
    TRACE();
    return ctx_destroy(ctx);
}

FWD_API CUresult cuCtxDestroy(CUcontext ctx) {
    TRACE();
    return ctx_destroy(ctx);
}

FWD_API CUresult cuCtxGetCurrent(CUcontext *ctx) {
    TRACE();
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    *ctx = current_ctx();
    return CUDA_SUCCESS;
}

FWD_API CUresult cuCtxSetCurrent(CUcontext ctx) {
    TRACE();
    if (g_ctx_depth) g_ctx_depth--;
    return ctx ? ctx_push(ctx) : CUDA_SUCCESS;
}

FWD_API CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    TRACE();
    return ctx_push(ctx);
}

FWD_API CUresult cuCtxPushCurrent(CUcontext ctx) {
    TRACE();
    return ctx_push(ctx);
}

static CUresult ctx_pop(CUcontext *ctx) {
    if (!g_ctx_depth) return CUDA_ERROR_INVALID_CONTEXT;
    CUcontext top = g_ctx_stack[--g_ctx_depth];
    if (ctx) *ctx = top;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuCtxPopCurrent_v2(CUcontext *ctx) {
    TRACE();
    return ctx_pop(ctx);
}

FWD_API CUresult cuCtxPopCurrent(CUcontext *ctx) {
    TRACE();
    return ctx_pop(ctx);
}

FWD_API CUresult cuCtxGetDevice(CUdevice *device) {
    TRACE();
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (!current_ctx()) return CUDA_ERROR_INVALID_CONTEXT;
    *device = g_ctx_device;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    TRACE();
    (void)ctx;
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    *version = 3020; /* the API level the 10.1 header selects by default */
    return CUDA_SUCCESS;
}

FWD_API CUresult cuCtxSynchronize(void) {
    TRACE();
    mp_buf req;
    req_begin(&req, "sync", 0);
    return rpc_void(&req);
}

FWD_API CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev) {
    TRACE();
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    if (!g_primary_ctx) {
        CUresult rc = ctx_create(&g_primary_ctx, 0, dev);
        if (rc != CUDA_SUCCESS) return rc;
    }
    g_primary_refs++;
    *ctx = g_primary_ctx;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    TRACE();
    (void)dev;
    if (!g_primary_ctx || g_primary_refs <= 0) return CUDA_ERROR_INVALID_CONTEXT;
    if (--g_primary_refs > 0) return CUDA_SUCCESS;
    CUcontext ctx = g_primary_ctx;
    g_primary_ctx = NULL;
    return ctx_destroy(ctx);
}

/* ---- shared host segments ------------------------------------------------ */

/* Memory from cuMemAllocHost / cuMemHostAlloc is a segment the server maps
 * too. A copy between it and the device sends only {handle, offset, len}; the
 * bytes never travel through the socket. */
#define FWD_MAX_REGIONS 64

typedef struct {
    uint8_t *base;
    size_t len;
    uint64_t handle;
} fwd_region;

static pthread_mutex_t g_region_lock = PTHREAD_MUTEX_INITIALIZER;
static fwd_region g_regions[FWD_MAX_REGIONS];

/* Find the segment that wholly contains [p, p + bytes). */
static int region_find(const void *p, size_t bytes, uint64_t *handle, uint64_t *offset) {
    int found = 0;
    const uint8_t *q = (const uint8_t *)p;
    pthread_mutex_lock(&g_region_lock);
    for (int i = 0; i < FWD_MAX_REGIONS; i++) {
        const fwd_region *r = &g_regions[i];
        if (r->base && q >= r->base && bytes <= r->len && (size_t)(q - r->base) <= r->len - bytes) {
            *handle = r->handle;
            *offset = (uint64_t)(q - r->base);
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_region_lock);
    return found;
}

/* Ask the server for a segment and map it. Returns NULL when shared host
 * memory is unavailable; the caller then falls back to ordinary memory. */
static void *region_alloc(size_t bytes) {
    pthread_mutex_lock(&g_lock);
    void *base = NULL;
    uint64_t handle = 0, size = 0;
    int fd = -1;
    if (connect_locked() == CUDA_SUCCESS && g_shared_host) {
        mp_buf req;
        req_begin(&req, "host_alloc", 1);
        arg_uint(&req, "size", bytes);
        fwd_reply reply;
        if (exchange_locked(&req, &reply) == CUDA_SUCCESS) {
            mp_cur cur;
            if (mp_find(reply.result, "handle", &cur)) handle = (uint64_t)mp_read_int(&cur);
            if (mp_find(reply.result, "size", &cur)) size = (uint64_t)mp_read_int(&cur);
            /* The descriptor follows every successful host_alloc reply and must
             * be taken off the socket before the next request. */
            fd = fwd_recv_fd(g_fd);
        }
        reply_free(&reply);
    }
    pthread_mutex_unlock(&g_lock);
    if (!handle) {
        if (fd >= 0) close(fd);
        return NULL;
    }
    if (fd >= 0) {
        if (size >= bytes) {
            void *mapped = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapped != MAP_FAILED) base = mapped;
        }
        close(fd);
    }
    if (base) {
        int stored = 0;
        pthread_mutex_lock(&g_region_lock);
        for (int i = 0; i < FWD_MAX_REGIONS && !stored; i++) {
            if (!g_regions[i].base) {
                g_regions[i].base = (uint8_t *)base;
                g_regions[i].len = (size_t)size;
                g_regions[i].handle = handle;
                stored = 1;
            }
        }
        pthread_mutex_unlock(&g_region_lock);
        if (stored) return base;
        munmap(base, (size_t)size);
    }
    /* Could not use the segment: give it back. */
    mp_buf req;
    req_begin(&req, "host_free", 1);
    arg_uint(&req, "handle", handle);
    (void)rpc_void(&req);
    return NULL;
}

/* Returns 1 when `p` was a shared segment (now released), 0 otherwise. */
static int region_free(void *p) {
    uint64_t handle = 0;
    size_t len = 0;
    pthread_mutex_lock(&g_region_lock);
    for (int i = 0; i < FWD_MAX_REGIONS; i++) {
        if (g_regions[i].base == (uint8_t *)p) {
            handle = g_regions[i].handle;
            len = g_regions[i].len;
            memset(&g_regions[i], 0, sizeof(g_regions[i]));
            break;
        }
    }
    pthread_mutex_unlock(&g_region_lock);
    if (!len) return 0;
    munmap(p, len);
    mp_buf req;
    req_begin(&req, "host_free", 1);
    arg_uint(&req, "handle", handle);
    (void)rpc_void(&req);
    return 1;
}

static CUresult copy_shared(const char *op, CUdeviceptr dptr, uint64_t handle, uint64_t offset, size_t bytes) {
    mp_buf req;
    req_begin(&req, op, 4);
    arg_uint(&req, "dptr", dptr);
    arg_uint(&req, "handle", handle);
    arg_uint(&req, "offset", offset);
    arg_uint(&req, "len", bytes);
    return rpc_void(&req);
}

/* ---- device memory ------------------------------------------------------ */

static CUresult mem_alloc(CUdeviceptr *dptr, uint64_t bytes) {
    if (!dptr || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    mp_buf req;
    req_begin(&req, "mem_alloc", 1);
    arg_uint(&req, "size", bytes);
    uint64_t base = 0;
    CUresult rc = rpc_u64(&req, &base);
    if (rc == CUDA_SUCCESS) *dptr = base;
    return rc;
}

static CUresult mem_free(CUdeviceptr dptr) {
    mp_buf req;
    req_begin(&req, "mem_free", 1);
    arg_uint(&req, "dptr", dptr);
    return rpc_void(&req);
}

static CUresult copy_h2d(CUdeviceptr dst, const void *src, size_t bytes) {
    if (!src && bytes) return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 0) return CUDA_SUCCESS;
    uint64_t handle, shared_offset;
    if (region_find(src, bytes, &handle, &shared_offset)) return copy_shared("h2d_shm", dst, handle, shared_offset, bytes);
    const uint8_t *p = (const uint8_t *)src;
    for (size_t offset = 0; offset < bytes;) {
        size_t n = bytes - offset < g_chunk ? bytes - offset : g_chunk;
        mp_buf req;
        req_begin(&req, "h2d", 4);
        arg_uint(&req, "dptr", dst);
        arg_uint(&req, "total", bytes);
        arg_uint(&req, "offset", offset);
        mp_str(&req, "data");
        mp_bin(&req, p + offset, n);
        CUresult rc = rpc_void(&req);
        if (rc != CUDA_SUCCESS) return rc;
        offset += n;
    }
    return CUDA_SUCCESS;
}

static CUresult copy_d2h(void *dst, CUdeviceptr src, size_t bytes) {
    if (!dst && bytes) return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 0) return CUDA_SUCCESS;
    uint64_t handle, shared_offset;
    if (region_find(dst, bytes, &handle, &shared_offset)) return copy_shared("d2h_shm", src, handle, shared_offset, bytes);
    uint8_t *p = (uint8_t *)dst;
    for (size_t offset = 0; offset < bytes;) {
        size_t n = bytes - offset < g_chunk ? bytes - offset : g_chunk;
        mp_buf req;
        req_begin(&req, "d2h", 4);
        arg_uint(&req, "dptr", src);
        arg_uint(&req, "total", bytes);
        arg_uint(&req, "offset", offset);
        arg_uint(&req, "len", n);
        fwd_reply reply;
        CUresult rc = rpc(&req, &reply);
        if (rc == CUDA_SUCCESS) {
            size_t got = 0;
            const uint8_t *data = reply.has_result ? mp_read_bytes(&reply.result, &got, 0) : NULL;
            if (!data || got != n) rc = CUDA_ERROR_UNKNOWN;
            else memcpy(p + offset, data, n);
        }
        reply_free(&reply);
        if (rc != CUDA_SUCCESS) return rc;
        offset += n;
    }
    return CUDA_SUCCESS;
}

/* Fill `count` elements of `width` bytes with `value` by uploading a pattern. */
static CUresult mem_set(CUdeviceptr dst, uint32_t value, size_t width, size_t count) {
    if (count == 0) return CUDA_SUCCESS;
    if (count > SIZE_MAX / width) return CUDA_ERROR_INVALID_VALUE;
    size_t bytes = count * width;
    uint8_t *pattern = (uint8_t *)malloc(bytes);
    if (!pattern) return CUDA_ERROR_OUT_OF_MEMORY;
    for (size_t i = 0; i < count; i++) memcpy(pattern + i * width, &value, width); /* little-endian hosts */
    CUresult rc = copy_h2d(dst, pattern, bytes);
    free(pattern);
    return rc;
}

static CUresult copy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    if (bytes == 0) return CUDA_SUCCESS;
    void *staging = malloc(bytes);
    if (!staging) return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult rc = copy_d2h(staging, src, bytes);
    if (rc == CUDA_SUCCESS) rc = copy_h2d(dst, staging, bytes);
    free(staging);
    return rc;
}

static CUresult mem_info(size_t *free_bytes, size_t *total_bytes) {
    mp_buf req;
    req_begin(&req, "mem_info", 0);
    fwd_reply reply;
    CUresult rc = rpc(&req, &reply);
    if (rc == CUDA_SUCCESS) {
        mp_cur cur;
        if (mp_find(reply.result, "free", &cur)) *free_bytes = (size_t)mp_read_int(&cur);
        if (mp_find(reply.result, "total", &cur)) *total_bytes = (size_t)mp_read_int(&cur);
    }
    reply_free(&reply);
    return rc;
}

FWD_API CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytes) {
    TRACE();
    return mem_alloc(dptr, bytes);
}

FWD_API CUresult cuMemFree_v2(CUdeviceptr dptr) {
    TRACE();
    return mem_free(dptr);
}

FWD_API CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes) {
    TRACE();
    return copy_h2d(dst, src, bytes);
}

FWD_API CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes) {
    TRACE();
    return copy_d2h(dst, src, bytes);
}

FWD_API CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    TRACE();
    return copy_d2d(dst, src, bytes);
}

/* The served stack completes every copy before it replies, so the Async
 * variants are the synchronous copy; the stream argument is not consulted. */
FWD_API CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream) {
    TRACE();
    (void)stream;
    return copy_h2d(dst, src, bytes);
}

FWD_API CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes, CUstream stream) {
    TRACE();
    (void)stream;
    return copy_d2h(dst, src, bytes);
}

FWD_API CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count) {
    TRACE();
    return mem_set(dst, value, 1, count);
}

FWD_API CUresult cuMemsetD16_v2(CUdeviceptr dst, unsigned short value, size_t count) {
    TRACE();
    return mem_set(dst, value, 2, count);
}

FWD_API CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count) {
    TRACE();
    return mem_set(dst, value, 4, count);
}

FWD_API CUresult cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes) {
    TRACE();
    if (!free_bytes || !total_bytes) return CUDA_ERROR_INVALID_VALUE;
    return mem_info(free_bytes, total_bytes);
}

/* Un-suffixed names: CUDA 3.x ABI on x86_64, the 64-bit ABI on arm64. */

FWD_API CUresult cuMemAlloc(fwd_dptr1 *dptr, fwd_size1 bytes) {
    TRACE();
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    CUdeviceptr base = 0;
    CUresult rc = mem_alloc(&base, bytes);
    if (rc != CUDA_SUCCESS) return rc;
#if FWD_LEGACY_V1
    if (base + bytes > 0x100000000ull) {
        /* A 32-bit CUdeviceptr cannot name this allocation. */
        (void)mem_free(base);
        fwd_note("cuMemAlloc (CUDA 3.x ABI) got a device address above 4 GiB; use cuMemAlloc_v2");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
#endif
    *dptr = (fwd_dptr1)base;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuMemFree(fwd_dptr1 dptr) {
    TRACE();
    return mem_free(dptr);
}

FWD_API CUresult cuMemcpyHtoD(fwd_dptr1 dst, const void *src, fwd_size1 bytes) {
    TRACE();
    return copy_h2d(dst, src, bytes);
}

FWD_API CUresult cuMemcpyDtoH(void *dst, fwd_dptr1 src, fwd_size1 bytes) {
    TRACE();
    return copy_d2h(dst, src, bytes);
}

FWD_API CUresult cuMemcpyDtoD(fwd_dptr1 dst, fwd_dptr1 src, fwd_size1 bytes) {
    TRACE();
    return copy_d2d(dst, src, bytes);
}

FWD_API CUresult cuMemsetD8(fwd_dptr1 dst, unsigned char value, fwd_size1 count) {
    TRACE();
    return mem_set(dst, value, 1, count);
}

FWD_API CUresult cuMemsetD32(fwd_dptr1 dst, unsigned int value, fwd_size1 count) {
    TRACE();
    return mem_set(dst, value, 4, count);
}

FWD_API CUresult cuMemGetInfo(fwd_size1 *free_bytes, fwd_size1 *total_bytes) {
    TRACE();
    if (!free_bytes || !total_bytes) return CUDA_ERROR_INVALID_VALUE;
    size_t f = 0, t = 0;
    CUresult rc = mem_info(&f, &t);
    if (rc != CUDA_SUCCESS) return rc;
#if FWD_LEGACY_V1
    *free_bytes = f > 0xffffffffull ? 0xffffffffu : (unsigned int)f;
    *total_bytes = t > 0xffffffffull ? 0xffffffffu : (unsigned int)t;
#else
    *free_bytes = f;
    *total_bytes = t;
#endif
    return CUDA_SUCCESS;
}

/* ---- host memory -------------------------------------------------------- */

/* Host memory from the CUDA allocators is a shared segment when the server
 * offers one, and ordinary page-aligned memory otherwise (copies then travel
 * over the socket). 16 KiB alignment suits both page sizes a Rosetta or
 * native process can see. Set CUDA_ADAPTER_NO_SHARED_HOST=1 to force ordinary
 * memory. */
#define FWD_HOST_ALIGN 16384u

static CUresult host_alloc(void **out, size_t bytes) {
    if (!out || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    void *p = region_alloc(bytes);
    if (p) {
        *out = p;
        return CUDA_SUCCESS;
    }
    size_t rounded = (bytes + FWD_HOST_ALIGN - 1) / FWD_HOST_ALIGN * FWD_HOST_ALIGN;
    if (rounded < bytes || posix_memalign(&p, FWD_HOST_ALIGN, rounded) != 0) return CUDA_ERROR_OUT_OF_MEMORY;
    *out = p;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuMemAllocHost_v2(void **out, size_t bytes) {
    TRACE();
    return host_alloc(out, bytes);
}

FWD_API CUresult cuMemAllocHost(void **out, fwd_size1 bytes) {
    TRACE();
    return host_alloc(out, bytes);
}

FWD_API CUresult cuMemHostAlloc(void **out, size_t bytes, unsigned int flags) {
    TRACE();
    (void)flags;
    return host_alloc(out, bytes);
}

FWD_API CUresult cuMemFreeHost(void *p) {
    TRACE();
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    if (!region_free(p)) free(p);
    return CUDA_SUCCESS;
}

/* ---- modules ------------------------------------------------------------ */

static uint64_t rd64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static uint32_t rd32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint16_t rd16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

/* cuModuleLoadData gives a pointer and no length. Derive the length from the
 * image itself: ELF tables, the fat binary header, or the PTX C string. */
static size_t image_size(const uint8_t *image) {
    if (memcmp(image, "\x7f" "ELF", 4) == 0) {
        if (image[4] != 2 || image[5] != 1) return 0; /* ELF64 little-endian only */
        uint64_t end = 64;
        uint64_t phoff = rd64(image + 32), shoff = rd64(image + 40);
        uint16_t phentsize = rd16(image + 54), phnum = rd16(image + 56);
        uint16_t shentsize = rd16(image + 58), shnum = rd16(image + 60);
        if (phnum && phentsize >= 56) {
            if (phoff + (uint64_t)phnum * phentsize > end) end = phoff + (uint64_t)phnum * phentsize;
            for (uint16_t i = 0; i < phnum; i++) {
                const uint8_t *ph = image + phoff + (uint64_t)i * phentsize;
                uint64_t stop = rd64(ph + 8) + rd64(ph + 32); /* p_offset + p_filesz */
                if (stop > end) end = stop;
            }
        }
        if (shnum && shentsize >= 64) {
            if (shoff + (uint64_t)shnum * shentsize > end) end = shoff + (uint64_t)shnum * shentsize;
            for (uint16_t i = 0; i < shnum; i++) {
                const uint8_t *sh = image + shoff + (uint64_t)i * shentsize;
                if (rd32(sh + 4) == 8) continue; /* SHT_NOBITS occupies no file bytes */
                uint64_t stop = rd64(sh + 24) + rd64(sh + 32); /* sh_offset + sh_size */
                if (stop > end) end = stop;
            }
        }
        return end <= FWD_MAX_FRAME ? (size_t)end : 0;
    }
    if (rd32(image) == 0xBA55ED50u) {
        uint64_t total = (uint64_t)rd16(image + 6) + rd64(image + 8); /* header + payload */
        return total <= FWD_MAX_FRAME ? (size_t)total : 0;
    }
    return strlen((const char *)image) + 1;
}

static void copy_log(char *dst, size_t cap, const char *text) {
    if (dst && cap) snprintf(dst, cap, "%s", text);
}

static CUresult module_load(CUmodule *module, const void *image, size_t bytes, int opt_level, char *info_log,
                            size_t info_cap, char *error_log, size_t error_cap) {
    if (!module || !image || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    mp_buf req;
    req_begin(&req, "module_load", opt_level >= 0 ? 2 : 1);
    mp_str(&req, "image");
    mp_bin(&req, image, bytes);
    if (opt_level >= 0) arg_uint(&req, "opt_level", (uint64_t)opt_level);
    fwd_reply reply;
    CUresult rc = rpc(&req, &reply);
    if (rc == CUDA_SUCCESS) {
        mp_cur cur;
        uint64_t handle = 0;
        if (mp_find(reply.result, "module", &cur)) handle = (uint64_t)mp_read_int(&cur);
        if (!handle) rc = CUDA_ERROR_UNKNOWN;
        else *module = (CUmodule)(uintptr_t)handle;
        if (info_log && info_cap && mp_find(reply.result, "log", &cur)) {
            size_t n;
            const uint8_t *text = mp_read_bytes(&cur, &n, 1);
            if (text) {
                if (n >= info_cap) n = info_cap - 1;
                memcpy(info_log, text, n);
                info_log[n] = 0;
            }
        }
    } else {
        copy_log(error_log, error_cap, reply.error);
        if (reply.error[0]) fwd_note("module load failed: %s", reply.error);
    }
    reply_free(&reply);
    return rc;
}

FWD_API CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    TRACE();
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    size_t bytes = image_size((const uint8_t *)image);
    if (!bytes) return CUDA_ERROR_INVALID_IMAGE;
    return module_load(module, image, bytes, -1, NULL, 0, NULL, 0);
}

FWD_API CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int count, int *options,
                                    void **values) {
    TRACE();
    if (!module || !image || (count && (!options || !values))) return CUDA_ERROR_INVALID_VALUE;
    int opt_level = -1;
    char *info_log = NULL, *error_log = NULL;
    size_t info_cap = 0, error_cap = 0;
    for (unsigned int i = 0; i < count; i++) {
        switch (options[i]) {
        case CU_JIT_OPTIMIZATION_LEVEL: opt_level = (int)(uintptr_t)values[i]; break;
        case CU_JIT_INFO_LOG_BUFFER: info_log = (char *)values[i]; break;
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: info_cap = (size_t)(uintptr_t)values[i]; break;
        case CU_JIT_ERROR_LOG_BUFFER: error_log = (char *)values[i]; break;
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: error_cap = (size_t)(uintptr_t)values[i]; break;
        default: break; /* target and cache options do not apply: the target is always the served card */
        }
    }
    if (opt_level > 4) return CUDA_ERROR_INVALID_VALUE;
    if (info_log && info_cap) info_log[0] = 0;
    if (error_log && error_cap) error_log[0] = 0;
    size_t bytes = image_size((const uint8_t *)image);
    if (!bytes) return CUDA_ERROR_INVALID_IMAGE;
    return module_load(module, image, bytes, opt_level, info_log, info_cap, error_log, error_cap);
}

FWD_API CUresult cuModuleLoadFatBinary(CUmodule *module, const void *image) {
    TRACE();
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    size_t bytes = image_size((const uint8_t *)image);
    if (!bytes) return CUDA_ERROR_INVALID_IMAGE;
    return module_load(module, image, bytes, -1, NULL, 0, NULL, 0);
}

FWD_API CUresult cuModuleLoad(CUmodule *module, const char *path) {
    TRACE();
    if (!module || !path) return CUDA_ERROR_INVALID_VALUE;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return CUDA_ERROR_FILE_NOT_FOUND;
    struct stat st;
    CUresult rc = CUDA_ERROR_INVALID_IMAGE;
    if (fstat(fd, &st) == 0 && st.st_size > 0 && (uint64_t)st.st_size <= FWD_MAX_FRAME) {
        size_t bytes = (size_t)st.st_size;
        uint8_t *image = (uint8_t *)malloc(bytes);
        if (!image) rc = CUDA_ERROR_OUT_OF_MEMORY;
        else if (fwd_read_all(fd, image, bytes) == 0) rc = module_load(module, image, bytes, -1, NULL, 0, NULL, 0);
        free(image);
    }
    close(fd);
    return rc;
}

FWD_API CUresult cuModuleUnload(CUmodule module) {
    TRACE();
    if (!module) return CUDA_ERROR_INVALID_HANDLE;
    mp_buf req;
    req_begin(&req, "module_unload", 1);
    arg_uint(&req, "module", (uint64_t)(uintptr_t)module);
    return rpc_void(&req);
}

/* Function handles are small local records. They live until the process
 * exits: CUDA has no call that frees one, and a module's functions must not
 * dangle if the application keeps a handle past cuModuleUnload. */
FWD_API CUresult cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name) {
    TRACE();
    if (!function || !module || !name) return CUDA_ERROR_INVALID_VALUE;
    mp_buf req;
    req_begin(&req, "function_get", 2);
    arg_uint(&req, "module", (uint64_t)(uintptr_t)module);
    mp_str(&req, "name");
    mp_str(&req, name);
    fwd_reply reply;
    CUresult rc = rpc(&req, &reply);
    if (rc == CUDA_SUCCESS) {
        fwd_function *fn = (fwd_function *)calloc(1, sizeof(*fn));
        mp_cur cur, sizes, offsets;
        if (!fn) {
            rc = CUDA_ERROR_OUT_OF_MEMORY;
        } else if (!mp_find(reply.result, "function", &cur) || !mp_find(reply.result, "param_sizes", &sizes) ||
                   !mp_find(reply.result, "param_offsets", &offsets)) {
            rc = CUDA_ERROR_UNKNOWN;
        } else {
            fn->remote = (uint64_t)mp_read_int(&cur);
            uint32_t n = mp_read_container(&sizes, 0);
            uint32_t m = mp_read_container(&offsets, 0);
            if (n != m || n > FWD_MAX_PARAMS || sizes.bad || offsets.bad) {
                rc = CUDA_ERROR_NOT_SUPPORTED; /* more parameters than the adapter handles */
            } else {
                fn->nparams = n;
                for (uint32_t i = 0; i < n; i++) {
                    fn->sizes[i] = (uint32_t)mp_read_int(&sizes);
                    fn->offsets[i] = (uint32_t)mp_read_int(&offsets);
                }
                fn->block[0] = fn->block[1] = fn->block[2] = 1;
                if (sizes.bad || offsets.bad) rc = CUDA_ERROR_UNKNOWN;
            }
        }
        if (rc == CUDA_SUCCESS) *function = (CUfunction)fn;
        else free(fn);
    }
    reply_free(&reply);
    return rc;
}

/* ---- launch ------------------------------------------------------------- */

/* Send a launch. `param(i)` yields the address of parameter i's bytes. */
static CUresult launch(fwd_function *fn, const unsigned int grid[3], const unsigned int block[3],
                       unsigned int shared, void **separate, const uint8_t *packed, size_t packed_len) {
    mp_buf req;
    req_begin(&req, "launch", 5);
    arg_uint(&req, "function", fn->remote);
    mp_str(&req, "grid");
    mp_array(&req, 3);
    for (int i = 0; i < 3; i++) mp_uint(&req, grid[i]);
    mp_str(&req, "block");
    mp_array(&req, 3);
    for (int i = 0; i < 3; i++) mp_uint(&req, block[i]);
    arg_uint(&req, "shared_mem_bytes", shared);
    mp_str(&req, "params");
    mp_array(&req, fn->nparams);
    for (uint32_t i = 0; i < fn->nparams; i++) {
        const uint8_t *src;
        if (separate) {
            src = (const uint8_t *)separate[i];
        } else if ((size_t)fn->offsets[i] + fn->sizes[i] <= packed_len) {
            src = packed + fn->offsets[i];
        } else {
            src = NULL;
        }
        if (!src) {
            free(req.data);
            return CUDA_ERROR_INVALID_VALUE;
        }
        mp_bin(&req, src, fn->sizes[i]);
    }
    return rpc_void(&req);
}

FWD_API CUresult cuLaunchKernel(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                                unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                CUstream stream, void **params, void **extra) {
    TRACE();
    (void)stream; /* launches complete before the reply; see cuStreamSynchronize */
    fwd_function *fn = (fwd_function *)function;
    if (!fn) return CUDA_ERROR_INVALID_HANDLE;
    unsigned int grid[3] = {gx, gy, gz}, block[3] = {bx, by, bz};
    if (params && extra) return CUDA_ERROR_INVALID_VALUE; /* one parameter form at a time */
    if (params || !fn->nparams) return launch(fn, grid, block, shared, params, NULL, 0);
    /* Packed form: {BUFFER_POINTER, ptr, BUFFER_SIZE, &size, END}. */
    const uint8_t *buffer = NULL;
    size_t size = 0;
    int have_size = 0;
    if (!extra) return CUDA_ERROR_INVALID_VALUE;
    for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
        if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buffer = (const uint8_t *)extra[i + 1];
        else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[i + 1]) {
            size = *(const size_t *)extra[i + 1];
            have_size = 1;
        } else return CUDA_ERROR_INVALID_VALUE;
    }
    if (!buffer || !have_size) return CUDA_ERROR_INVALID_VALUE;
    return launch(fn, grid, block, shared, NULL, buffer, size);
}

/* CUDA 3.x launch protocol: the parameter block is built with cuParamSet*
 * at explicit offsets, then launched with cuLaunch / cuLaunchGrid. */

static CUresult legacy_set(CUfunction function, int offset, const void *src, size_t bytes) {
    fwd_function *fn = (fwd_function *)function;
    if (!fn) return CUDA_ERROR_INVALID_HANDLE;
    if (offset < 0 || !src || (size_t)offset + bytes > FWD_LEGACY_PARAM_BYTES) return CUDA_ERROR_INVALID_VALUE;
    memcpy(fn->legacy + offset, src, bytes);
    return CUDA_SUCCESS;
}

FWD_API CUresult cuParamSetSize(CUfunction function, unsigned int bytes) {
    TRACE();
    fwd_function *fn = (fwd_function *)function;
    if (!fn) return CUDA_ERROR_INVALID_HANDLE;
    if (bytes > FWD_LEGACY_PARAM_BYTES) return CUDA_ERROR_INVALID_VALUE;
    fn->legacy_size = bytes;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuParamSeti(CUfunction function, int offset, unsigned int value) {
    TRACE();
    return legacy_set(function, offset, &value, sizeof(value));
}

FWD_API CUresult cuParamSetf(CUfunction function, int offset, float value) {
    TRACE();
    return legacy_set(function, offset, &value, sizeof(value));
}

FWD_API CUresult cuParamSetv(CUfunction function, int offset, void *src, unsigned int bytes) {
    TRACE();
    return legacy_set(function, offset, src, bytes);
}

FWD_API CUresult cuFuncSetBlockShape(CUfunction function, int x, int y, int z) {
    TRACE();
    fwd_function *fn = (fwd_function *)function;
    if (!fn) return CUDA_ERROR_INVALID_HANDLE;
    if (x <= 0 || y <= 0 || z <= 0) return CUDA_ERROR_INVALID_VALUE;
    fn->block[0] = (uint32_t)x;
    fn->block[1] = (uint32_t)y;
    fn->block[2] = (uint32_t)z;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuFuncSetSharedSize(CUfunction function, unsigned int bytes) {
    TRACE();
    fwd_function *fn = (fwd_function *)function;
    if (!fn) return CUDA_ERROR_INVALID_HANDLE;
    fn->shared = bytes;
    return CUDA_SUCCESS;
}

static CUresult legacy_launch(CUfunction function, int width, int height) {
    fwd_function *fn = (fwd_function *)function;
    if (!fn) return CUDA_ERROR_INVALID_HANDLE;
    if (width <= 0 || height <= 0) return CUDA_ERROR_INVALID_VALUE;
    unsigned int grid[3] = {(unsigned int)width, (unsigned int)height, 1};
    return launch(fn, grid, fn->block, fn->shared, NULL, fn->legacy, fn->legacy_size);
}

FWD_API CUresult cuLaunch(CUfunction function) {
    TRACE();
    return legacy_launch(function, 1, 1);
}

FWD_API CUresult cuLaunchGrid(CUfunction function, int width, int height) {
    TRACE();
    return legacy_launch(function, width, height);
}

FWD_API CUresult cuLaunchGridAsync(CUfunction function, int width, int height, CUstream stream) {
    TRACE();
    (void)stream;
    return legacy_launch(function, width, height);
}

/* ---- streams and events ------------------------------------------------- */

/* The served stack runs one launch at a time and each call returns after the
 * work is done, so a stream is only an identity here and is always idle. */

FWD_API CUresult cuStreamCreate(CUstream *stream, unsigned int flags) {
    TRACE();
    (void)flags;
    if (!stream) return CUDA_ERROR_INVALID_VALUE;
    *stream = (CUstream)calloc(1, 8);
    return *stream ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

FWD_API CUresult cuStreamDestroy_v2(CUstream stream) {
    TRACE();
    free(stream);
    return CUDA_SUCCESS;
}

FWD_API CUresult cuStreamDestroy(CUstream stream) {
    TRACE();
    free(stream);
    return CUDA_SUCCESS;
}

FWD_API CUresult cuStreamSynchronize(CUstream stream) {
    TRACE();
    (void)stream;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuStreamQuery(CUstream stream) {
    TRACE();
    (void)stream;
    return CUDA_SUCCESS;
}

/* Events record the HOST clock at cuEventRecord. They are not GPU timestamps:
 * an elapsed time includes the socket round trip and must not be reported as
 * device time. */

FWD_API CUresult cuEventCreate(CUevent *event, unsigned int flags) {
    TRACE();
    (void)flags;
    if (!event) return CUDA_ERROR_INVALID_VALUE;
    *event = (CUevent)calloc(1, sizeof(fwd_event));
    return *event ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

FWD_API CUresult cuEventRecord(CUevent event, CUstream stream) {
    TRACE();
    (void)stream;
    fwd_event *e = (fwd_event *)event;
    if (!e) return CUDA_ERROR_INVALID_HANDLE;
    clock_gettime(CLOCK_MONOTONIC, &e->at);
    e->recorded = 1;
    return CUDA_SUCCESS;
}

FWD_API CUresult cuEventQuery(CUevent event) {
    TRACE();
    return event ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

FWD_API CUresult cuEventSynchronize(CUevent event) {
    TRACE();
    return event ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

FWD_API CUresult cuEventElapsedTime(float *milliseconds, CUevent start, CUevent end) {
    TRACE();
    fwd_event *a = (fwd_event *)start, *b = (fwd_event *)end;
    if (!milliseconds) return CUDA_ERROR_INVALID_VALUE;
    if (!a || !b) return CUDA_ERROR_INVALID_HANDLE;
    if (!a->recorded || !b->recorded) return CUDA_ERROR_INVALID_HANDLE;
    *milliseconds = (float)((double)(b->at.tv_sec - a->at.tv_sec) * 1e3 + (double)(b->at.tv_nsec - a->at.tv_nsec) / 1e6);
    return CUDA_SUCCESS;
}

FWD_API CUresult cuEventDestroy_v2(CUevent event) {
    TRACE();
    free(event);
    return CUDA_SUCCESS;
}

FWD_API CUresult cuEventDestroy(CUevent event) {
    TRACE();
    free(event);
    return CUDA_SUCCESS;
}

/* ---- error text --------------------------------------------------------- */

static const struct {
    CUresult code;
    const char *name;
    const char *text;
} k_errors[] = {
    {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
    {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
    {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
    {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
    {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"},
    {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
    {CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid"},
    {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
    {CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU", "no kernel image is available for execution on the device"},
    {CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND", "file not found"},
    {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"},
    {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
    {CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready"},
    {CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure"},
    {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported"},
    {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error"},
};

FWD_API CUresult cuGetErrorName(CUresult error, const char **text) {
    TRACE();
    if (!text) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof(k_errors) / sizeof(k_errors[0]); i++)
        if (k_errors[i].code == error) {
            *text = k_errors[i].name;
            return CUDA_SUCCESS;
        }
    *text = NULL;
    return CUDA_ERROR_INVALID_VALUE;
}

FWD_API CUresult cuGetErrorString(CUresult error, const char **text) {
    TRACE();
    if (!text) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof(k_errors) / sizeof(k_errors[0]); i++)
        if (k_errors[i].code == error) {
            *text = k_errors[i].text;
            return CUDA_SUCCESS;
        }
    *text = NULL;
    return CUDA_ERROR_INVALID_VALUE;
}
