/*
 * shims/shim_dma.c -- Linux DMA KPI for the nvidia-ko translator.
 *
 * KOT_SAFE_SHIMS variant (RM closure link, shims/macos_os, host tests):
 *   - a mutex around the shadow list (live-allocation registry); allocation
 *     only for the device the identity gate accepted (kot_device.h);
 *   - DMA QUARANTINE: dma_free_coherent never calls cudaDriverKitKpiDmaFree
 *     unless kot_device_quiesced() (shims/macos_os/kot_lifecycle.h). Otherwise
 *     the buffer stays allocated and mapped on a quarantine list, the free
 *     returns to RM as done, and kot_quarantine_report() lists it;
 *   - a double free (free of a quarantined buffer) and a free of an unknown
 *     pointer are reported (stderr once per kind, counted), never acted on;
 *   - dma_unmap_single never frees; map/unmap/sync on a quarantined buffer is
 *     refused and counted.
 * Without KOT_SAFE_SHIMS (default 570.190 / 595.71.05 build): the legacy code
 * below the #else, kept byte-identical; it frees immediately and is NOT fit to
 * be put in front of a card.
 */
#if defined(KOT_SAFE_SHIMS)
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_driverkit_kpi.h>
#include <linux/dma-mapping.h>
#include <shim_mutation_admission.h>

#include "macos_os/kot_device.h"
#include "macos_os/kot_lifecycle.h"

#define SHIM_DMA_MIN_ALIGN 256ULL
#define SHIM_DMA_LOW32_LIMIT (1ULL << 32)

enum { SHIM_DMA_LIVE = 1, SHIM_DMA_QUARANTINED = 2 };

struct shim_dma_shadow {
    uint64_t handle;
    void *host_addr;
    uint64_t device_addr;
    uint64_t requested_size;
    uint64_t actual_size;
    uint64_t map_count;
    uint64_t unmap_count;
    uint64_t sync_to_device_count;
    uint64_t sync_from_device_count;
    uint64_t last_sync_device_addr;
    uint64_t last_sync_size;
    enum dma_data_direction last_direction;
    uint32_t state;
    const void *free_caller;
    struct shim_dma_shadow *next;
};

struct shim_dma_snapshot {
    uint64_t handle;
    uint64_t host_addr;
    uint64_t device_addr;
    uint64_t requested_size;
    uint64_t actual_size;
    uint64_t low32;
    uint64_t aligned_256;
    uint64_t map_count;
    uint64_t unmap_count;
    uint64_t sync_to_device_count;
    uint64_t sync_from_device_count;
    uint64_t last_sync_device_addr;
    uint64_t last_sync_size;
    uint32_t last_direction;
};

struct shim_dma_stats {
    uint64_t live, quarantined, kpi_freed;
    uint64_t double_free, unknown_free, size_mismatch, use_after_free;
};

static pthread_mutex_t g_dma_lock = PTHREAD_MUTEX_INITIALIZER;
static struct shim_dma_shadow *g_dma_shadows;
static struct shim_dma_stats g_dma_stats;
static int g_dma_once[4];

static void shim_dma_publish_locked(void)
{
    kot_lifecycle_dma_counts(g_dma_stats.live, g_dma_stats.quarantined);
}

static void shim_dma_report(int kind, uint64_t *ctr, const char *what, const void *p, const void *caller)
{
    pthread_mutex_lock(&g_dma_lock);
    (*ctr)++;
    int first = !g_dma_once[kind];
    g_dma_once[kind] = 1;
    pthread_mutex_unlock(&g_dma_lock);
    if (first)
        fprintf(stderr, "kot-dma: %s (pointer %p, caller %p); not acted on, counted (reported once per kind)\n",
                what, p, caller);
}

static int shim_dma_low32_range(uint64_t device_addr, uint64_t size)
{
    if (device_addr >= SHIM_DMA_LOW32_LIMIT) {
        return 0;
    }
    if (size > SHIM_DMA_LOW32_LIMIT - device_addr) {
        return 0;
    }
    return 1;
}

static int shim_dma_aligned(uint64_t device_addr)
{
    return (device_addr & (SHIM_DMA_MIN_ALIGN - 1ULL)) == 0;
}

/* all lookups: caller holds g_dma_lock */
static struct shim_dma_shadow *shim_dma_find_by_host(void *host_addr)
{
    for (struct shim_dma_shadow *it = g_dma_shadows; it != 0; it = it->next)
        if (it->host_addr == host_addr)
            return it;
    return 0;
}

static struct shim_dma_shadow *shim_dma_find_by_device(dma_addr_t device_addr)
{
    for (struct shim_dma_shadow *it = g_dma_shadows; it != 0; it = it->next)
        if (it->device_addr == (uint64_t)device_addr)
            return it;
    return 0;
}

static int shim_dma_range_contains(uint64_t base, uint64_t len, uint64_t addr, uint64_t size)
{
    if (addr < base) {
        return 0;
    }
    uint64_t offset = addr - base;
    if (offset > len) {
        return 0;
    }
    return size <= len - offset;
}

static struct shim_dma_shadow *shim_dma_find_by_device_range(dma_addr_t device_addr, size_t size)
{
    for (struct shim_dma_shadow *it = g_dma_shadows; it != 0; it = it->next)
        if (shim_dma_range_contains(it->device_addr, it->actual_size, (uint64_t)device_addr, (uint64_t)size))
            return it;
    return 0;
}

static void shim_dma_remove(struct shim_dma_shadow *shadow)
{
    struct shim_dma_shadow **slot = &g_dma_shadows;
    while (*slot != 0) {
        if (*slot == shadow) {
            *slot = shadow->next;
            shadow->next = 0;
            return;
        }
        slot = &(*slot)->next;
    }
}

void *dma_alloc_coherent(struct device *dev, size_t size, dma_addr_t *dma_handle, gfp_t gfp)
{
    (void)dev;
    (void)gfp;
    CudaDriverKitKpiDmaAlloc alloc = {0};
    /* DMA memory is mapped for the device: only for the accepted card. */
    if (size == 0 || !kot_device_ready() || shim_mutation_admit(CUDK_MUTATION_OP_DMA_ALLOC) != 0) {
        return 0;
    }
    if (cudaDriverKitKpiDmaAlloc((uint64_t)size, 1, &alloc) != 0) {
        return 0;
    }
    struct shim_dma_shadow *shadow = (struct shim_dma_shadow *)calloc(1, sizeof(*shadow));
    int bad = shadow == 0 || alloc.host_addr == 0 || alloc.device_addr == 0 || alloc.actual_size < (uint64_t)size ||
              !shim_dma_low32_range(alloc.device_addr, alloc.actual_size) || !shim_dma_aligned(alloc.device_addr);
    if (bad) {
        /* Never handed to RM: no engine was told this IOVA, so it is released. */
        (void)cudaDriverKitKpiDmaFree(alloc.handle);
        free(shadow);
        return 0;
    }
    shadow->handle = alloc.handle;
    shadow->host_addr = (void *)(uintptr_t)alloc.host_addr;
    shadow->device_addr = alloc.device_addr;
    shadow->requested_size = (uint64_t)size;
    shadow->actual_size = alloc.actual_size;
    shadow->last_direction = DMA_NONE;
    shadow->state = SHIM_DMA_LIVE;
    /* Linux dma_alloc_coherent returns zeroed memory; the KPI does not promise it. */
    memset(shadow->host_addr, 0, (size_t)size);
    if (dma_handle != 0) {
        *dma_handle = alloc.device_addr;
    }
    pthread_mutex_lock(&g_dma_lock);
    shadow->next = g_dma_shadows;
    g_dma_shadows = shadow;
    g_dma_stats.live++;
    shim_dma_publish_locked();
    pthread_mutex_unlock(&g_dma_lock);
    return shadow->host_addr;
}

void dma_free_coherent(struct device *dev, size_t size, void *vaddr, dma_addr_t dma_handle)
{
    (void)dev;
    const void *caller = __builtin_return_address(0);
    if (vaddr == 0)
        return;
    pthread_mutex_lock(&g_dma_lock);
    struct shim_dma_shadow *shadow = shim_dma_find_by_host(vaddr);
    if (shadow == 0) {
        pthread_mutex_unlock(&g_dma_lock);
        shim_dma_report(1, &g_dma_stats.unknown_free, "dma_free_coherent of a pointer this layer did not allocate",
                        vaddr, caller);
        return;
    }
    if (shadow->state == SHIM_DMA_QUARANTINED) {
        pthread_mutex_unlock(&g_dma_lock);
        shim_dma_report(0, &g_dma_stats.double_free, "DOUBLE FREE: dma_free_coherent of a quarantined buffer",
                        vaddr, caller);
        return;
    }
    int mismatch = (dma_handle != 0 && (uint64_t)dma_handle != shadow->device_addr) ||
                   (size != 0 && (uint64_t)size > shadow->actual_size);
    if (kot_device_quiesced()) {
        shim_dma_remove(shadow);
        g_dma_stats.live--;
        g_dma_stats.kpi_freed++;
        shim_dma_publish_locked();
        pthread_mutex_unlock(&g_dma_lock);
        (void)cudaDriverKitKpiDmaFree(shadow->handle);
        free(shadow);
    } else {
        shadow->state = SHIM_DMA_QUARANTINED;
        shadow->free_caller = caller;
        g_dma_stats.live--;
        g_dma_stats.quarantined++;
        shim_dma_publish_locked();
        pthread_mutex_unlock(&g_dma_lock);
    }
    if (mismatch)
        shim_dma_report(2, &g_dma_stats.size_mismatch, "dma_free_coherent size/handle do not match the allocation",
                        vaddr, caller);
}

uint64_t kot_quarantine_report(FILE *out)
{
    if (out == 0)
        out = stderr;
    uint64_t n = 0;
    pthread_mutex_lock(&g_dma_lock);
    fprintf(out, "# kot DMA quarantine: %llu buffers kept allocated and mapped (device %s%s%s); live %llu\n",
            (unsigned long long)g_dma_stats.quarantined, kot_device_quiesced() ? "quiesced: " : "NOT quiesced",
            kot_device_quiesced() ? kot_device_quiesced_reason() : "", "", (unsigned long long)g_dma_stats.live);
    for (struct shim_dma_shadow *it = g_dma_shadows; it != 0; it = it->next) {
        if (it->state != SHIM_DMA_QUARANTINED)
            continue;
        n++;
        fprintf(out, "quarantined handle %llu host %p iova 0x%llx size %llu freed-by %p\n",
                (unsigned long long)it->handle, it->host_addr, (unsigned long long)it->device_addr,
                (unsigned long long)it->actual_size, it->free_caller);
    }
    pthread_mutex_unlock(&g_dma_lock);
    return n;
}

int64_t kot_quarantine_release(void)
{
    if (!kot_device_quiesced()) {
        fprintf(stderr, "kot-dma: kot_quarantine_release refused: the device is not quiesced\n");
        return -1;
    }
    int64_t n = 0;
    for (;;) {
        pthread_mutex_lock(&g_dma_lock);
        struct shim_dma_shadow *it = g_dma_shadows;
        while (it != 0 && it->state != SHIM_DMA_QUARANTINED)
            it = it->next;
        if (it != 0) {
            shim_dma_remove(it);
            g_dma_stats.quarantined--;
            g_dma_stats.kpi_freed++;
            shim_dma_publish_locked();
        }
        pthread_mutex_unlock(&g_dma_lock);
        if (it == 0)
            break;
        (void)cudaDriverKitKpiDmaFree(it->handle);
        free(it);
        n++;
    }
    return n;
}

int shim_dma_stats(struct shim_dma_stats *out, uint64_t out_size)
{
    if (out == 0 || out_size < sizeof(*out))
        return -1;
    pthread_mutex_lock(&g_dma_lock);
    *out = g_dma_stats;
    pthread_mutex_unlock(&g_dma_lock);
    return 0;
}

/* The buffer for a map/sync, or 0 (reported) if it is quarantined. */
static struct shim_dma_shadow *shim_dma_usable_locked(struct shim_dma_shadow *s, const void *p, const void *caller,
                                                      int *uaf)
{
    *uaf = 0;
    if (s != 0 && s->state == SHIM_DMA_QUARANTINED) {
        *uaf = 1;
        return 0;
    }
    (void)p;
    (void)caller;
    return s;
}

dma_addr_t dma_map_single(struct device *dev, void *ptr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    int uaf;
    int to_dev = dir == DMA_TO_DEVICE || dir == DMA_BIDIRECTIONAL;
    if (to_dev && shim_mutation_admit(CUDK_MUTATION_OP_DMA_SYNC_TO_DEVICE) != 0)
        return 0;
    pthread_mutex_lock(&g_dma_lock);
    struct shim_dma_shadow *shadow = shim_dma_usable_locked(shim_dma_find_by_host(ptr), ptr, 0, &uaf);
    uint64_t handle = 0, iova = 0;
    if (shadow != 0) {
        shadow->map_count++;
        shadow->last_direction = dir;
        if (to_dev) {
            shadow->sync_to_device_count++;
            shadow->last_sync_device_addr = shadow->device_addr;
            shadow->last_sync_size = (uint64_t)size;
        }
        handle = shadow->handle;
        iova = shadow->device_addr;
    }
    pthread_mutex_unlock(&g_dma_lock);
    if (uaf)
        shim_dma_report(3, &g_dma_stats.use_after_free, "map/sync of a quarantined (freed) buffer", ptr,
                        __builtin_return_address(0));
    if (shadow == 0)
        return 0;
    if (to_dev)
        (void)cudaDriverKitKpiDmaSyncToDevice(handle);
    return (dma_addr_t)iova;
}

void dma_unmap_single(struct device *dev, dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    int uaf;
    int from_dev = dir == DMA_FROM_DEVICE || dir == DMA_BIDIRECTIONAL;
    if (from_dev && shim_mutation_admit(CUDK_MUTATION_OP_DMA_SYNC_FROM_DEVICE) != 0)
        return;
    pthread_mutex_lock(&g_dma_lock);
    struct shim_dma_shadow *shadow = shim_dma_usable_locked(shim_dma_find_by_device(addr), 0, 0, &uaf);
    uint64_t handle = 0;
    if (shadow != 0) {
        shadow->unmap_count++;
        shadow->last_direction = dir;
        if (from_dev) {
            shadow->sync_from_device_count++;
            shadow->last_sync_device_addr = (uint64_t)addr;
            shadow->last_sync_size = (uint64_t)size;
        }
        handle = shadow->handle;
    }
    pthread_mutex_unlock(&g_dma_lock);
    if (uaf)
        shim_dma_report(3, &g_dma_stats.use_after_free, "unmap of a quarantined (freed) buffer",
                        (const void *)(uintptr_t)addr, __builtin_return_address(0));
    /* Unmapping never unpins: the buffer is released only by dma_free_coherent. */
    if (shadow != 0 && from_dev)
        (void)cudaDriverKitKpiDmaSyncFromDevice(handle);
}

int dma_mapping_error(struct device *dev, dma_addr_t dma_addr)
{
    (void)dev;
    pthread_mutex_lock(&g_dma_lock);
    struct shim_dma_shadow *s = dma_addr == 0 ? 0 : shim_dma_find_by_device(dma_addr);
    int err = s == 0 || s->state != SHIM_DMA_LIVE;
    pthread_mutex_unlock(&g_dma_lock);
    return err;
}

static void shim_dma_sync(dma_addr_t addr, size_t size, enum dma_data_direction dir, int to_device, const void *caller)
{
    int uaf;
    int want = to_device ? (dir == DMA_TO_DEVICE || dir == DMA_BIDIRECTIONAL)
                         : (dir == DMA_FROM_DEVICE || dir == DMA_BIDIRECTIONAL);
    if (!want)
        return;
    if (shim_mutation_admit(to_device ? CUDK_MUTATION_OP_DMA_SYNC_TO_DEVICE : CUDK_MUTATION_OP_DMA_SYNC_FROM_DEVICE) != 0)
        return;
    pthread_mutex_lock(&g_dma_lock);
    struct shim_dma_shadow *shadow = shim_dma_usable_locked(shim_dma_find_by_device_range(addr, size), 0, 0, &uaf);
    uint64_t handle = 0;
    if (shadow != 0) {
        if (to_device)
            shadow->sync_to_device_count++;
        else
            shadow->sync_from_device_count++;
        shadow->last_sync_device_addr = (uint64_t)addr;
        shadow->last_sync_size = (uint64_t)size;
        shadow->last_direction = dir;
        handle = shadow->handle;
    }
    pthread_mutex_unlock(&g_dma_lock);
    if (uaf)
        shim_dma_report(3, &g_dma_stats.use_after_free, "sync of a quarantined (freed) buffer",
                        (const void *)(uintptr_t)addr, caller);
    if (shadow == 0)
        return;
    if (to_device)
        (void)cudaDriverKitKpiDmaSyncToDevice(handle);
    else
        (void)cudaDriverKitKpiDmaSyncFromDevice(handle);
}

void dma_sync_single_for_device(struct device *dev, dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    shim_dma_sync(addr, size, dir, 1, __builtin_return_address(0));
}

void dma_sync_single_for_cpu(struct device *dev, dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    shim_dma_sync(addr, size, dir, 0, __builtin_return_address(0));
}

void dma_sync_single_range_for_device(struct device *dev, dma_addr_t addr, unsigned long offset, size_t size, enum dma_data_direction dir)
{
    dma_sync_single_for_device(dev, addr + (dma_addr_t)offset, size, dir);
}

void dma_sync_single_range_for_cpu(struct device *dev, dma_addr_t addr, unsigned long offset, size_t size, enum dma_data_direction dir)
{
    dma_sync_single_for_cpu(dev, addr + (dma_addr_t)offset, size, dir);
}

int dma_set_mask_and_coherent(struct device *dev, u64 mask)
{
    (void)dev;
    (void)mask;
    return 0;
}

int shim_dmaSnapshot(void *host_addr, struct shim_dma_snapshot *out, uint64_t out_size)
{
    if (out == 0 || out_size < sizeof(*out)) {
        return -1;
    }
    pthread_mutex_lock(&g_dma_lock);
    struct shim_dma_shadow *shadow = shim_dma_find_by_host(host_addr);
    if (shadow == 0) {
        pthread_mutex_unlock(&g_dma_lock);
        return -2;
    }
    out->handle = shadow->handle;
    out->host_addr = (uint64_t)(uintptr_t)shadow->host_addr;
    out->device_addr = shadow->device_addr;
    out->requested_size = shadow->requested_size;
    out->actual_size = shadow->actual_size;
    out->low32 = shim_dma_low32_range(shadow->device_addr, shadow->actual_size) ? 1 : 0;
    out->aligned_256 = shim_dma_aligned(shadow->device_addr) ? 1 : 0;
    out->map_count = shadow->map_count;
    out->unmap_count = shadow->unmap_count;
    out->sync_to_device_count = shadow->sync_to_device_count;
    out->sync_from_device_count = shadow->sync_from_device_count;
    out->last_sync_device_addr = shadow->last_sync_device_addr;
    out->last_sync_size = shadow->last_sync_size;
    out->last_direction = (uint32_t)shadow->last_direction;
    pthread_mutex_unlock(&g_dma_lock);
    return 0;
}

#else /* !KOT_SAFE_SHIMS: legacy default-build variant, unchanged */
#include <stdint.h>
#include <stdlib.h>

#include <cuda_driverkit_kpi.h>
#include <linux/dma-mapping.h>
#include <shim_mutation_admission.h>

#define SHIM_DMA_MIN_ALIGN 256ULL
#define SHIM_DMA_LOW32_LIMIT (1ULL << 32)

struct shim_dma_shadow {
    uint64_t handle;
    void *host_addr;
    uint64_t device_addr;
    uint64_t requested_size;
    uint64_t actual_size;
    uint64_t map_count;
    uint64_t unmap_count;
    uint64_t sync_to_device_count;
    uint64_t sync_from_device_count;
    uint64_t last_sync_device_addr;
    uint64_t last_sync_size;
    enum dma_data_direction last_direction;
    struct shim_dma_shadow *next;
};

struct shim_dma_snapshot {
    uint64_t handle;
    uint64_t host_addr;
    uint64_t device_addr;
    uint64_t requested_size;
    uint64_t actual_size;
    uint64_t low32;
    uint64_t aligned_256;
    uint64_t map_count;
    uint64_t unmap_count;
    uint64_t sync_to_device_count;
    uint64_t sync_from_device_count;
    uint64_t last_sync_device_addr;
    uint64_t last_sync_size;
    uint32_t last_direction;
};

static struct shim_dma_shadow *g_dma_shadows;

static int shim_dma_low32_range(uint64_t device_addr, uint64_t size)
{
    if (device_addr >= SHIM_DMA_LOW32_LIMIT) {
        return 0;
    }
    if (size > SHIM_DMA_LOW32_LIMIT - device_addr) {
        return 0;
    }
    return 1;
}

static int shim_dma_aligned(uint64_t device_addr)
{
    return (device_addr & (SHIM_DMA_MIN_ALIGN - 1ULL)) == 0;
}

static struct shim_dma_shadow *shim_dma_find_by_host(void *host_addr)
{
    struct shim_dma_shadow *it = g_dma_shadows;
    while (it != 0) {
        if (it->host_addr == host_addr) {
            return it;
        }
        it = it->next;
    }
    return 0;
}

static struct shim_dma_shadow *shim_dma_find_by_device(dma_addr_t device_addr)
{
    struct shim_dma_shadow *it = g_dma_shadows;
    while (it != 0) {
        if (it->device_addr == (uint64_t)device_addr) {
            return it;
        }
        it = it->next;
    }
    return 0;
}

static int shim_dma_range_contains(uint64_t base, uint64_t len, uint64_t addr, uint64_t size)
{
    if (addr < base) {
        return 0;
    }
    uint64_t offset = addr - base;
    if (offset > len) {
        return 0;
    }
    return size <= len - offset;
}

static struct shim_dma_shadow *shim_dma_find_by_device_range(dma_addr_t device_addr, size_t size)
{
    struct shim_dma_shadow *it = g_dma_shadows;
    uint64_t addr = (uint64_t)device_addr;
    while (it != 0) {
        if (shim_dma_range_contains(it->device_addr, it->actual_size, addr, (uint64_t)size)) {
            return it;
        }
        it = it->next;
    }
    return 0;
}

static void shim_dma_insert(struct shim_dma_shadow *shadow)
{
    shadow->next = g_dma_shadows;
    g_dma_shadows = shadow;
}

static void shim_dma_remove(struct shim_dma_shadow *shadow)
{
    struct shim_dma_shadow **slot = &g_dma_shadows;
    while (*slot != 0) {
        if (*slot == shadow) {
            *slot = shadow->next;
            shadow->next = 0;
            return;
        }
        slot = &(*slot)->next;
    }
}

void *dma_alloc_coherent(struct device *dev, size_t size, dma_addr_t *dma_handle, gfp_t gfp)
{
    (void)dev;
    (void)gfp;
    CudaDriverKitKpiDmaAlloc alloc = {0};
    if (shim_mutation_admit(CUDK_MUTATION_OP_DMA_ALLOC) != 0) {
        return 0;
    }
    if (cudaDriverKitKpiDmaAlloc((uint64_t)size, 1, &alloc) != 0) {
        return 0;
    }
    struct shim_dma_shadow *shadow = (struct shim_dma_shadow *)malloc(sizeof(*shadow));
    if (shadow == 0) {
        (void)cudaDriverKitKpiDmaFree(alloc.handle);
        return 0;
    }
    shadow->handle = alloc.handle;
    shadow->host_addr = (void *)(uintptr_t)alloc.host_addr;
    shadow->device_addr = alloc.device_addr;
    shadow->requested_size = (uint64_t)size;
    shadow->actual_size = alloc.actual_size;
    shadow->map_count = 0;
    shadow->unmap_count = 0;
    shadow->sync_to_device_count = 0;
    shadow->sync_from_device_count = 0;
    shadow->last_sync_device_addr = 0;
    shadow->last_sync_size = 0;
    shadow->last_direction = DMA_NONE;
    shadow->next = 0;
    if (dma_handle != 0) {
        *dma_handle = alloc.device_addr;
    }
    if (shadow->host_addr == 0 ||
        shadow->device_addr == 0 ||
        shadow->actual_size < (uint64_t)size ||
        !shim_dma_low32_range(shadow->device_addr, shadow->actual_size) ||
        !shim_dma_aligned(shadow->device_addr)) {
        (void)cudaDriverKitKpiDmaFree(alloc.handle);
        free(shadow);
        return 0;
    }
    shim_dma_insert(shadow);
    return shadow->host_addr;
}

void dma_free_coherent(struct device *dev, size_t size, void *vaddr, dma_addr_t dma_handle)
{
    (void)dev;
    (void)size;
    (void)dma_handle;
    struct shim_dma_shadow *shadow = shim_dma_find_by_host(vaddr);
    if (shadow != 0) {
        shim_dma_remove(shadow);
        (void)cudaDriverKitKpiDmaFree(shadow->handle);
        free(shadow);
    }
}

dma_addr_t dma_map_single(struct device *dev, void *ptr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    (void)size;
    struct shim_dma_shadow *shadow = shim_dma_find_by_host(ptr);
    if (shadow == 0) {
        return 0;
    }
    if (dir == DMA_TO_DEVICE || dir == DMA_BIDIRECTIONAL) {
        if (shim_mutation_admit(CUDK_MUTATION_OP_DMA_SYNC_TO_DEVICE) != 0) {
            return 0;
        }
    }
    shadow->map_count++;
    shadow->last_direction = dir;
    if (dir == DMA_TO_DEVICE || dir == DMA_BIDIRECTIONAL) {
        (void)cudaDriverKitKpiDmaSyncToDevice(shadow->handle);
        shadow->sync_to_device_count++;
        shadow->last_sync_device_addr = shadow->device_addr;
        shadow->last_sync_size = (uint64_t)size;
    }
    return (dma_addr_t)shadow->device_addr;
}

void dma_unmap_single(struct device *dev, dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    (void)size;
    struct shim_dma_shadow *shadow = shim_dma_find_by_device(addr);
    if (shadow != 0) {
        if (dir == DMA_FROM_DEVICE || dir == DMA_BIDIRECTIONAL) {
            if (shim_mutation_admit(CUDK_MUTATION_OP_DMA_SYNC_FROM_DEVICE) != 0) {
                return;
            }
        }
        shadow->unmap_count++;
        shadow->last_direction = dir;
        if (dir == DMA_FROM_DEVICE || dir == DMA_BIDIRECTIONAL) {
            (void)cudaDriverKitKpiDmaSyncFromDevice(shadow->handle);
            shadow->sync_from_device_count++;
            shadow->last_sync_device_addr = (uint64_t)addr;
            shadow->last_sync_size = (uint64_t)size;
        }
    }
}

int dma_mapping_error(struct device *dev, dma_addr_t dma_addr)
{
    (void)dev;
    return dma_addr == 0 || shim_dma_find_by_device(dma_addr) == 0;
}

void dma_sync_single_for_device(struct device *dev, dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    struct shim_dma_shadow *shadow = shim_dma_find_by_device_range(addr, size);
    if (shadow != 0 && (dir == DMA_TO_DEVICE || dir == DMA_BIDIRECTIONAL)) {
        if (shim_mutation_admit(CUDK_MUTATION_OP_DMA_SYNC_TO_DEVICE) != 0) {
            return;
        }
        (void)cudaDriverKitKpiDmaSyncToDevice(shadow->handle);
        shadow->sync_to_device_count++;
        shadow->last_sync_device_addr = (uint64_t)addr;
        shadow->last_sync_size = (uint64_t)size;
        shadow->last_direction = dir;
    }
}

void dma_sync_single_for_cpu(struct device *dev, dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    (void)dev;
    struct shim_dma_shadow *shadow = shim_dma_find_by_device_range(addr, size);
    if (shadow != 0 && (dir == DMA_FROM_DEVICE || dir == DMA_BIDIRECTIONAL)) {
        if (shim_mutation_admit(CUDK_MUTATION_OP_DMA_SYNC_FROM_DEVICE) != 0) {
            return;
        }
        (void)cudaDriverKitKpiDmaSyncFromDevice(shadow->handle);
        shadow->sync_from_device_count++;
        shadow->last_sync_device_addr = (uint64_t)addr;
        shadow->last_sync_size = (uint64_t)size;
        shadow->last_direction = dir;
    }
}

void dma_sync_single_range_for_device(struct device *dev, dma_addr_t addr, unsigned long offset, size_t size, enum dma_data_direction dir)
{
    dma_sync_single_for_device(dev, addr + (dma_addr_t)offset, size, dir);
}

void dma_sync_single_range_for_cpu(struct device *dev, dma_addr_t addr, unsigned long offset, size_t size, enum dma_data_direction dir)
{
    dma_sync_single_for_cpu(dev, addr + (dma_addr_t)offset, size, dir);
}

int dma_set_mask_and_coherent(struct device *dev, u64 mask)
{
    (void)dev;
    (void)mask;
    return 0;
}

int shim_dmaSnapshot(void *host_addr, struct shim_dma_snapshot *out, uint64_t out_size)
{
    if (out == 0 || out_size < sizeof(*out)) {
        return -1;
    }
    struct shim_dma_shadow *shadow = shim_dma_find_by_host(host_addr);
    if (shadow == 0) {
        return -2;
    }
    out->handle = shadow->handle;
    out->host_addr = (uint64_t)(uintptr_t)shadow->host_addr;
    out->device_addr = shadow->device_addr;
    out->requested_size = shadow->requested_size;
    out->actual_size = shadow->actual_size;
    out->low32 = shim_dma_low32_range(shadow->device_addr, shadow->actual_size) ? 1 : 0;
    out->aligned_256 = shim_dma_aligned(shadow->device_addr) ? 1 : 0;
    out->map_count = shadow->map_count;
    out->unmap_count = shadow->unmap_count;
    out->sync_to_device_count = shadow->sync_to_device_count;
    out->sync_from_device_count = shadow->sync_from_device_count;
    out->last_sync_device_addr = shadow->last_sync_device_addr;
    out->last_sync_size = shadow->last_sync_size;
    out->last_direction = (uint32_t)shadow->last_direction;
    return 0;
}
#endif /* KOT_SAFE_SHIMS */
