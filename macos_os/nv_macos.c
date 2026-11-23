/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/nv_macos.c -- macOS replacement for the RM-facing nv_*
 * services of kernel-open/nvidia/nv.c (OGK 570.190), plus the nv_* entry
 * points of the other Linux-only files the closure fails to compile
 * (nv-dma.c, nv-acpi.c, nv-nano-timer.c, libspdm_*.c), plus the handful of
 * nv.c globals and helpers that kernel-open/nvidia/nv-pci.c references.
 *
 * Device access: DMA through shim_dma.c (dma_alloc_coherent / dma_free_coherent
 * / dma_sync_single_for_cpu -> cudaDriverKitKpiDmaAlloc / DmaSyncFromDevice);
 * the device IOVA is the address RM sees. nv_free_pages never releases the
 * DMA buffer itself: dma_free_coherent quarantines it unless the device is
 * quiesced (kot_lifecycle.h), and only then calls cudaDriverKitKpiDmaFree. Firmware through
 * shim_firmware.c (request_firmware under $NVIDIA_KO_FIRMWARE_ROOT).
 * nv_is_gpu_accessible() asks cudaDriverKitKpiProbe (read-only).
 * Everything else is refused (stderr note once, NV_ERR_NOT_SUPPORTED).
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef KOT_OGK_VERSION_STRING
#define KOT_OGK_VERSION_STRING "570.190"
#endif
/* Same file names as nv.c: "nvidia/<version>/gsp_*.bin" (shim_firmware maps them). */
#define NV_FIRMWARE_FOR_NAME(name) "nvidia/" KOT_OGK_VERSION_STRING "/" name ".bin"
#include "nvtypes.h"
#include "nvstatus.h"
#include "os-interface.h"
#include "nv.h"
#include "nv-firmware.h"   /* nv.h includes it; NV_FIRMWARE_FOR_NAME is defined above */


#include <cuda_driverkit_kpi.h>
#include <linux/dma-mapping.h>
#include <linux/firmware.h>
#include <linux/pci.h>
#include <linux/semaphore.h>
#include <shim_mutation_admission.h>

#include <pthread.h>

#include "macos_os_priv.h"
#include "kot_lifecycle.h"
#include "kot_device.h"

/* ======================================================================== */
/* DMA-backed page allocations (nv_alloc_pages family)                       */
/* ======================================================================== */
typedef struct kot_alloc {
    NvU32 magic;         /* KOT_ALLOC_MAGIC while registered */
    void *host;          /* CPU address from cudaDriverKitKpiDmaAlloc */
    dma_addr_t iova;     /* device address: what RM sees as "physical" */
    size_t size;
    NvU32 page_count;
    NvBool contiguous;
    struct kot_alloc *next;
} kot_alloc_t;
#define KOT_ALLOC_MAGIC 0x6b6f7461u   /* 'kota' */

/* Registry of live nv_alloc_pages allocations: nv_free_pages frees only what it
 * finds here, so a double free or a foreign pointer is reported, not acted on. */
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;
static kot_alloc_t *g_allocs;
static uint64_t g_alloc_bad_free;
static int g_alloc_bad_once;

static void kot_alloc_register(kot_alloc_t *at)
{
    pthread_mutex_lock(&g_alloc_lock);
    at->magic = KOT_ALLOC_MAGIC;
    at->next = g_allocs;
    g_allocs = at;
    pthread_mutex_unlock(&g_alloc_lock);
}

/* Unlink at if registered; 1 if it was. */
static int kot_alloc_unregister(kot_alloc_t *at)
{
    int found = 0;
    pthread_mutex_lock(&g_alloc_lock);
    for (kot_alloc_t **pp = &g_allocs; *pp != NULL; pp = &(*pp)->next) {
        if (*pp == at) {
            *pp = at->next;
            at->next = NULL;
            at->magic = 0;
            found = 1;
            break;
        }
    }
    if (!found)
        g_alloc_bad_free++;
    int first = !found && !g_alloc_bad_once;
    if (!found)
        g_alloc_bad_once = 1;
    pthread_mutex_unlock(&g_alloc_lock);
    if (first)
        fprintf(stderr, "kot-macos-os: nv_free_pages of %p, which is not a live nv_alloc_pages allocation "
                        "(double free or foreign pointer); not acted on, counted (reported once)\n", (void *)at);
    return found;
}

uint64_t kot_alloc_bad_free_count(void)
{
    pthread_mutex_lock(&g_alloc_lock);
    uint64_t n = g_alloc_bad_free;
    pthread_mutex_unlock(&g_alloc_lock);
    return n;
}

NV_STATUS NV_API_CALL nv_alloc_pages(nv_state_t *nv, NvU32 page_count, NvU64 page_size,
                                     NvBool contiguous, NvU32 cache_type, NvBool zeroed,
                                     NvBool unencrypted, NvS32 node_id, NvU64 *pte_array,
                                     void **priv_data)
{
    (void)nv; (void)page_size; (void)cache_type; (void)node_id;
    if (page_count == 0 || pte_array == NULL || priv_data == NULL)
        return NV_ERR_INVALID_ARGUMENT;
    if (unencrypted) {
        KOT_REFUSED();   /* confidential-compute shared pages */
        return NV_ERR_NOT_SUPPORTED;
    }
    kot_alloc_t *at = calloc(1, sizeof(*at));
    if (at == NULL)
        return NV_ERR_NO_MEMORY;
    at->size = (size_t)page_count * KOT_OS_PAGE_SIZE;
    at->page_count = page_count;
    at->contiguous = contiguous;
    /* One KPI DMA buffer; contiguous in IOVA, so it serves both layouts. */
    at->host = dma_alloc_coherent(NULL, at->size, &at->iova, 0);
    if (at->host == NULL) {
        free(at);
        return NV_ERR_NO_MEMORY;
    }
    if (zeroed)
        memset(at->host, 0, at->size);
    for (NvU32 i = 0; i < (contiguous ? 1u : page_count); i++)
        pte_array[i] = (NvU64)at->iova + (NvU64)i * KOT_OS_PAGE_SIZE;
    kot_alloc_register(at);
    *priv_data = at;
    return NV_OK;
}

NV_STATUS NV_API_CALL nv_free_pages(nv_state_t *nv, NvU32 page_count, NvBool contiguous,
                                    NvU32 cache_type, void *priv_data)
{
    (void)nv; (void)page_count; (void)contiguous; (void)cache_type;
    kot_alloc_t *at = priv_data;
    if (at == NULL)
        return NV_ERR_INVALID_ARGUMENT;
    if (!kot_alloc_unregister(at))
        return NV_ERR_INVALID_ARGUMENT;
    /* Quarantined (kept allocated and mapped) unless the device is quiesced. */
    dma_free_coherent(NULL, at->size, at->host, at->iova);
    free(at);
    return NV_OK;
}

void *NV_API_CALL nv_alloc_kernel_mapping(nv_state_t *nv, void *pAllocPrivate, NvU64 pageIndex,
                                          NvU32 pageOffset, NvU64 size, void **pPrivate)
{
    (void)nv;
    kot_alloc_t *at = pAllocPrivate;
    if (at == NULL)
        return NULL;
    NvU64 off = pageIndex * KOT_OS_PAGE_SIZE + pageOffset;
    if (off > at->size || size > at->size - off)
        return NULL;
    if (pPrivate) *pPrivate = NULL;
    return (char *)at->host + off;
}

NV_STATUS NV_API_CALL nv_free_kernel_mapping(nv_state_t *nv, void *pAllocPrivate, void *address, void *pPrivate)
{
    (void)nv; (void)pAllocPrivate; (void)address; (void)pPrivate;
    return NV_OK;   /* the CPU mapping lives as long as the allocation */
}

/* IOVA == RM "physical" address, so mapping for DMA is the identity. */
NV_STATUS NV_API_CALL nv_dma_map_alloc(nv_dma_device_t *dma_dev, NvU64 page_count, NvU64 *va_array,
                                       NvBool contig, void **priv)
{
    (void)dma_dev; (void)page_count; (void)va_array; (void)contig; (void)priv;
    return NV_OK;
}

NV_STATUS NV_API_CALL nv_dma_unmap_alloc(nv_dma_device_t *dma_dev, NvU64 page_count, NvU64 *va_array, void **priv)
{
    (void)dma_dev; (void)page_count; (void)va_array; (void)priv;
    return NV_OK;
}

/* priv is the nv_alloc_pages() allocation: make device writes visible to the CPU. */
void NV_API_CALL nv_dma_cache_invalidate(nv_dma_device_t *dma_dev, void *priv)
{
    (void)dma_dev;
    kot_alloc_t *at = priv;
    if (at != NULL)
        dma_sync_single_for_cpu(NULL, at->iova, at->size, DMA_FROM_DEVICE);
}

NV_STATUS NV_API_CALL nv_dma_map_mmio(nv_dma_device_t *dma_dev, NvU64 page_count, NvU64 *va)
{
    (void)dma_dev; (void)page_count; (void)va;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_dma_unmap_mmio(nv_dma_device_t *dma_dev, NvU64 page_count, NvU64 va)
{
    (void)dma_dev; (void)page_count; (void)va;
    KOT_REFUSED();
}
NV_STATUS NV_API_CALL nv_dma_map_peer(nv_dma_device_t *dma_dev, nv_dma_device_t *peer_dma_dev,
                                      NvU8 bar_index, NvU64 page_count, NvU64 *va)
{
    (void)dma_dev; (void)peer_dma_dev; (void)bar_index; (void)page_count; (void)va;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_dma_unmap_peer(nv_dma_device_t *dma_dev, NvU64 page_count, NvU64 va)
{
    (void)dma_dev; (void)page_count; (void)va;
    KOT_REFUSED();
}
NV_STATUS NV_API_CALL nv_dma_import_sgt(nv_dma_device_t *dma_dev, struct sg_table *sgt, struct drm_gem_object *gem)
{
    (void)dma_dev; (void)sgt; (void)gem;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_dma_release_sgt(struct sg_table *sgt, struct drm_gem_object *gem)
{
    (void)sgt; (void)gem;
    KOT_REFUSED();
}

/* Barrier only: userspace cannot clean/invalidate by physical range (stub). */
void NV_API_CALL nv_flush_coherent_cpu_cache_range(nv_state_t *nv, NvU64 cpu_virtual, NvU64 size)
{
    (void)nv; (void)cpu_virtual; (void)size;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __builtin_arm_dsb(0xf);
}

/* ======================================================================== */
/* firmware (shim_firmware.c)                                                */
/* ======================================================================== */
const void *NV_API_CALL nv_get_firmware(nv_state_t *nv, nv_firmware_type_t fw_type,
                                        nv_firmware_chip_family_t fw_chip_family,
                                        const void **fw_buf, NvU32 *fw_size)
{
    (void)nv;
    const struct firmware *fw = NULL;
    const char *name = nv_firmware_for_chip_family(fw_type, fw_chip_family);
    if (name == NULL || name[0] == '\0')
        return NULL;
    if (request_firmware(&fw, name, NULL) != 0 || fw == NULL)
        return NULL;
    *fw_size = (NvU32)fw->size;
    *fw_buf = fw->data;
    return fw;
}

void NV_API_CALL nv_put_firmware(const void *fw_handle)
{
    release_firmware(fw_handle);
}

/* ======================================================================== */
/* device state queries                                                      */
/* ======================================================================== */
static nv_state_t g_ctl_state;
nv_state_t *NV_API_CALL nv_get_ctl_state(void) { return &g_ctl_state; }

NvBool NV_API_CALL nv_is_gpu_accessible(nv_state_t *nv)
{
    (void)nv;
    uint16_t vendor = 0, device = 0;
    uint32_t max_selector = 0;
    if (!kot_device_ready())
        return NV_FALSE;   /* identity gate (kot_device.h) */
    return (cudaDriverKitKpiProbe(&vendor, &device, &max_selector) == 0 && vendor == 0x10de &&
            device == kot_device_pci_device()) ? NV_TRUE : NV_FALSE;
}

static NvU32 g_dma_address_size = 64;
void NV_API_CALL nv_set_dma_address_size(nv_state_t *nv, NvU32 phys_addr_bits)
{
    (void)nv;
    g_dma_address_size = phys_addr_bits;   /* KPI DMA is allocated low32 anyway */
}

NvBool NV_API_CALL nv_requires_dma_remap(nv_state_t *nv) { (void)nv; return NV_FALSE; }
NvU32 NV_API_CALL nv_get_dev_minor(nv_state_t *nv) { (void)nv; return 0; }   /* stub: one device */
NvBool NV_API_CALL nv_is_chassis_notebook(void) { return NV_FALSE; }
NvBool NV_API_CALL nv_platform_supports_s0ix(void) { return NV_FALSE; }
NvBool NV_API_CALL nv_s2idle_pm_configured(void) { return NV_FALSE; }
NvBool NV_API_CALL nv_dynamic_power_available(nv_state_t *nv) { (void)nv; return NV_FALSE; }

NV_STATUS NV_API_CALL nv_log_error(nv_state_t *nv, NvU32 error_number, const char *format, va_list ap)
{
    (void)nv;
    fprintf(stderr, "NVRM: Xid %u: ", error_number);
    vfprintf(stderr, format, ap);
    return NV_OK;
}

/* ======================================================================== */
/* interrupts: record + poll (no MSI); model in macos_os_priv.h              */
/* ======================================================================== */
static pthread_mutex_t g_isr_lock = PTHREAD_MUTEX_INITIALIZER;
static kot_isr_fn g_isr;
static kot_bh_fn g_bh;
static void *g_isr_ctx;
static uint64_t g_isr_skipped;

int kot_irq_register(kot_isr_fn isr, kot_bh_fn bh, void *ctx)
{
    pthread_mutex_lock(&g_isr_lock);
    g_isr = isr;
    g_bh = bh;
    g_isr_ctx = ctx;
    pthread_mutex_unlock(&g_isr_lock);
    return 0;
}

void kot_irq_unregister(void)
{
    pthread_mutex_lock(&g_isr_lock);
    g_isr = NULL;
    g_bh = NULL;
    g_isr_ctx = NULL;
    pthread_mutex_unlock(&g_isr_lock);
}

int kot_poll_interrupts(void)
{
    pthread_mutex_lock(&g_isr_lock);
    kot_isr_fn isr = g_isr;
    kot_bh_fn bh = g_bh;
    void *ctx = g_isr_ctx;
    pthread_mutex_unlock(&g_isr_lock);
    if (isr == NULL)
        return -1;
    if (kot_lifecycle_fatal_entered())
        return -2;
    /* exclusive against every os_acquire_spinlock holder (10 ms, then skip) */
    if (!kot_irq_gate_enter_isr(10000000ull)) {
        pthread_mutex_lock(&g_isr_lock);
        g_isr_skipped++;
        pthread_mutex_unlock(&g_isr_lock);
        return -3;
    }
    int need_bh = 0;
    int handled = isr(ctx, &need_bh);
    kot_irq_gate_exit_isr();
    if (need_bh && bh != NULL)
        bh(ctx);
    return handled ? 1 : 0;
}

/* Optional periodic poll thread (not started by default). */
static pthread_mutex_t g_poll_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_poll_cv = PTHREAD_COND_INITIALIZER;
static int g_poll_running, g_poll_stop;
static uint64_t g_poll_period_ns, g_poll_cycles;
static pthread_t g_poll_thread;

static void *kot_irq_poll_main(void *arg)
{
    (void)arg;
    pthread_mutex_lock(&g_poll_lock);
    while (!g_poll_stop) {
        struct timespec rel = { (time_t)(g_poll_period_ns / 1000000000ull), (long)(g_poll_period_ns % 1000000000ull) };
        pthread_cond_timedwait_relative_np(&g_poll_cv, &g_poll_lock, &rel);
        if (g_poll_stop)
            break;
        pthread_mutex_unlock(&g_poll_lock);
        if (!kot_lifecycle_fatal_entered())
            (void)kot_poll_interrupts();
        pthread_mutex_lock(&g_poll_lock);
        g_poll_cycles++;
    }
    pthread_mutex_unlock(&g_poll_lock);
    return NULL;
}

int kot_irq_poll_thread_start(uint64_t period_us)
{
    if (period_us == 0)
        return -1;
    pthread_mutex_lock(&g_poll_lock);
    if (g_poll_running) {
        pthread_mutex_unlock(&g_poll_lock);
        return -1;
    }
    g_poll_stop = 0;
    g_poll_period_ns = period_us * 1000ull;
    int rc = pthread_create(&g_poll_thread, NULL, kot_irq_poll_main, NULL);
    g_poll_running = rc == 0;
    pthread_mutex_unlock(&g_poll_lock);
    return rc == 0 ? 0 : -1;
}

void kot_irq_poll_thread_stop(void)
{
    pthread_mutex_lock(&g_poll_lock);
    int running = g_poll_running;
    g_poll_stop = 1;
    pthread_cond_broadcast(&g_poll_cv);
    pthread_mutex_unlock(&g_poll_lock);
    if (running)
        pthread_join(g_poll_thread, NULL);
    pthread_mutex_lock(&g_poll_lock);
    g_poll_running = 0;
    pthread_mutex_unlock(&g_poll_lock);
}

uint64_t kot_irq_poll_cycles(void)
{
    pthread_mutex_lock(&g_poll_lock);
    uint64_t n = g_poll_cycles;
    pthread_mutex_unlock(&g_poll_lock);
    return n;
}

/* ======================================================================== */
/* refused: events/files, power management, display, timers, UVM, user and   */
/* peer memory registration, fabric, ACPI, confidential compute              */
/* ======================================================================== */
void NV_API_CALL nv_post_event(nv_event_t *event, NvHandle hParent, NvU32 index, NvU32 info32,
                               NvU16 info16, NvBool data_valid)
{
    (void)event; (void)hParent; (void)index; (void)info32; (void)info16; (void)data_valid;
    KOT_REFUSED();
}
NvS32 NV_API_CALL nv_get_event(nv_file_private_t *nvfp, nv_event_t *event, NvU32 *pending)
{
    (void)nvfp; (void)event;
    KOT_REFUSED();
    if (pending) *pending = 0;
    return NV_ERR_NOT_SUPPORTED;
}
nv_file_private_t *NV_API_CALL nv_get_file_private(NvS32 fd, NvBool ctl, void **os_private)
{
    (void)fd; (void)ctl;
    KOT_REFUSED();
    if (os_private) *os_private = NULL;
    return NULL;
}
void NV_API_CALL nv_put_file_private(void *os_private) { (void)os_private; }

void NV_API_CALL nv_allow_runtime_suspend(nv_state_t *nv) { (void)nv; KOT_REFUSED(); }
void NV_API_CALL nv_disallow_runtime_suspend(nv_state_t *nv) { (void)nv; KOT_REFUSED(); }
void NV_API_CALL nv_idle_holdoff(nv_state_t *nv) { (void)nv; KOT_REFUSED(); }
NV_STATUS NV_API_CALL nv_indicate_idle(nv_state_t *nv) { (void)nv; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
NV_STATUS NV_API_CALL nv_indicate_not_idle(nv_state_t *nv) { (void)nv; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
void NV_API_CALL nv_audio_dynamic_power(nv_state_t *nv) { (void)nv; KOT_REFUSED(); }
void NV_API_CALL nv_control_soc_irqs(nv_state_t *nv, NvBool bEnable) { (void)nv; (void)bEnable; KOT_REFUSED(); }

void NV_API_CALL nv_get_screen_info(nv_state_t *nv, NvU64 *pPhysicalAddress, NvU32 *pFbWidth,
                                    NvU32 *pFbHeight, NvU32 *pFbDepth, NvU32 *pFbPitch, NvU64 *pFbSize)
{
    (void)nv;
    KOT_REFUSED();
    if (pPhysicalAddress) *pPhysicalAddress = 0;
    if (pFbWidth) *pFbWidth = 0;
    if (pFbHeight) *pFbHeight = 0;
    if (pFbDepth) *pFbDepth = 0;
    if (pFbPitch) *pFbPitch = 0;
    if (pFbSize) *pFbSize = 0;
}
NV_STATUS NV_API_CALL nv_set_primary_vga_status(nv_state_t *nv) { (void)nv; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
void NV_API_CALL nv_get_disp_smmu_stream_ids(nv_state_t *nv, NvU32 *dispIsoStreamId, NvU32 *dispNisoStreamId)
{
    (void)nv;
    KOT_REFUSED();
    if (dispIsoStreamId) *dispIsoStreamId = NV_U32_MAX;
    if (dispNisoStreamId) *dispNisoStreamId = NV_U32_MAX;
}
NvBool NV_API_CALL nv_match_gpu_os_info(nv_state_t *nv, void *os_info) { (void)nv; (void)os_info; KOT_REFUSED(); return NV_FALSE; }

NV_STATUS NV_API_CALL nv_get_device_memory_config(nv_state_t *nv, NvU64 *compr_addr_sys_phys,
                                                  NvU64 *addr_guest_phys, NvU64 *rsvd_phys,
                                                  NvU32 *addr_width, NvS32 *node_id)
{
    (void)nv; (void)compr_addr_sys_phys; (void)addr_guest_phys; (void)rsvd_phys; (void)addr_width; (void)node_id;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_get_egm_info(nv_state_t *nv, NvU64 *phys_addr, NvU64 *size, NvS32 *egm_node_id)
{
    (void)nv; (void)phys_addr; (void)size; (void)egm_node_id;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}

/* nv_start_rc_timer / nv_stop_rc_timer and the nv_*_nano_timer family:
 * kot_timer_macos.c (one timer thread, callbacks delivered on the worker). */

void NV_API_CALL nv_schedule_uvm_isr(nv_state_t *nv) { (void)nv; KOT_REFUSED(); }
NV_STATUS NV_API_CALL nv_schedule_uvm_drain_p2p(NvU8 *pUuid) { (void)pUuid; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
void NV_API_CALL nv_schedule_uvm_resume_p2p(NvU8 *pUuid) { (void)pUuid; KOT_REFUSED(); }

NV_STATUS NV_API_CALL nv_register_user_pages(nv_state_t *nv, NvU64 page_count, NvU64 *phys_addr,
                                             void *import_priv, void **priv_data)
{
    (void)nv; (void)page_count; (void)phys_addr; (void)import_priv;
    KOT_REFUSED();
    if (priv_data) *priv_data = NULL;
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_unregister_user_pages(nv_state_t *nv, NvU64 page_count, void **import_priv, void **priv_data)
{
    (void)nv; (void)page_count; (void)import_priv; (void)priv_data;
    KOT_REFUSED();
}
NV_STATUS NV_API_CALL nv_register_peer_io_mem(nv_state_t *nv, NvU64 *phys_addr, NvU64 page_count, void **priv_data)
{
    (void)nv; (void)phys_addr; (void)page_count;
    KOT_REFUSED();
    if (priv_data) *priv_data = NULL;
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_unregister_peer_io_mem(nv_state_t *nv, void *priv_data) { (void)nv; (void)priv_data; KOT_REFUSED(); }
NV_STATUS NV_API_CALL nv_register_phys_pages(nv_state_t *nv, NvU64 *phys_addr, NvU64 page_count,
                                             NvU32 cache_type, void **priv_data)
{
    (void)nv; (void)phys_addr; (void)page_count; (void)cache_type;
    KOT_REFUSED();
    if (priv_data) *priv_data = NULL;
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_unregister_phys_pages(nv_state_t *nv, void *priv_data) { (void)nv; (void)priv_data; KOT_REFUSED(); }
NV_STATUS NV_API_CALL nv_register_sgt(nv_state_t *nv, NvU64 *phys_addr, NvU64 page_count, NvU32 cache_type,
                                      void **priv_data, struct sg_table *import_sgt, void *import_priv)
{
    (void)nv; (void)phys_addr; (void)page_count; (void)cache_type; (void)import_sgt; (void)import_priv;
    KOT_REFUSED();
    if (priv_data) *priv_data = NULL;
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL nv_unregister_sgt(nv_state_t *nv, struct sg_table **import_sgt, void **import_priv, void *priv_data)
{
    (void)nv; (void)import_sgt; (void)import_priv; (void)priv_data;
    KOT_REFUSED();
}
NV_STATUS NV_API_CALL nv_alias_pages(nv_state_t *nv, NvU32 page_cnt, NvU64 page_size, NvU32 contiguous,
                                     NvU32 cache_type, NvU64 guest_id, NvU64 *pte_array, NvBool carveout,
                                     void **priv_data)
{
    (void)nv; (void)page_cnt; (void)page_size; (void)contiguous; (void)cache_type; (void)guest_id;
    (void)pte_array; (void)carveout;
    KOT_REFUSED();
    if (priv_data) *priv_data = NULL;
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_get_num_phys_pages(void *pAllocPrivate, NvU32 *pNumPages)
{
    (void)pAllocPrivate; (void)pNumPages;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_get_phys_pages(void *pAllocPrivate, void *pPages, NvU32 *pNumPages)
{
    (void)pAllocPrivate; (void)pPages; (void)pNumPages;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_acquire_fabric_mgmt_cap(int fd, int *duped_fd)
{
    (void)fd; (void)duped_fd;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}

/* ACPI (nv-acpi.c) */
NV_STATUS NV_API_CALL nv_acpi_ddc_method(nv_state_t *nv, void *pEdidBuffer, NvU32 *pSize, NvBool bReadMultiBlock)
{
    (void)nv; (void)pEdidBuffer; (void)pSize; (void)bReadMultiBlock;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_acpi_dod_method(nv_state_t *nv, NvU32 *pOutData, NvU32 *pSize)
{
    (void)nv; (void)pOutData; (void)pSize;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_acpi_dsm_method(nv_state_t *nv, NvU8 *pAcpiDsmGuid, NvU32 acpiDsmRev,
                                         NvBool acpiNvpcfDsmFunction, NvU32 acpiDsmSubFunction,
                                         void *pInParams, NvU16 inParamSize, NvU32 *outStatus,
                                         void *pOutData, NvU16 *pSize)
{
    (void)nv; (void)pAcpiDsmGuid; (void)acpiDsmRev; (void)acpiNvpcfDsmFunction; (void)acpiDsmSubFunction;
    (void)pInParams; (void)inParamSize; (void)outStatus; (void)pOutData; (void)pSize;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_acpi_mux_method(nv_state_t *nv, NvU32 *pInOut, NvU32 muxAcpiId, const char *pMethodName)
{
    (void)nv; (void)pInOut; (void)muxAcpiId; (void)pMethodName;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_acpi_rom_method(nv_state_t *nv, NvU32 *pInData, NvU32 *pOutData)
{
    (void)nv; (void)pInData; (void)pOutData;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL nv_acpi_get_powersource(NvU32 *ac_plugged)
{
    (void)ac_plugged;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NvBool NV_API_CALL nv_acpi_is_battery_present(void) { KOT_REFUSED(); return NV_FALSE; }
void NV_API_CALL nv_acpi_methods_init(NvU32 *handlesPresent)
{
    KOT_REFUSED();
    if (handlesPresent) *handlesPresent = 0;
}
void NV_API_CALL nv_acpi_methods_uninit(void) { KOT_REFUSED(); }

/* libspdm crypto glue (kernel-open/nvidia/libspdm_*.c): confidential compute. */
bool libspdm_aead_gcm_prealloc(void **context)
{
    KOT_REFUSED();
    if (context) *context = NULL;
    return false;
}
void libspdm_aead_free(void *context) { (void)context; KOT_REFUSED(); }
bool libspdm_aead_aes_gcm_encrypt_prealloc(void *context, const uint8_t *key, size_t key_size,
                                           const uint8_t *iv, size_t iv_size, const uint8_t *a_data,
                                           size_t a_data_size, const uint8_t *data_in, size_t data_in_size,
                                           uint8_t *tag_out, size_t tag_size, uint8_t *data_out,
                                           size_t *data_out_size)
{
    (void)context; (void)key; (void)key_size; (void)iv; (void)iv_size; (void)a_data; (void)a_data_size;
    (void)data_in; (void)data_in_size; (void)tag_out; (void)tag_size; (void)data_out; (void)data_out_size;
    KOT_REFUSED();
    return false;
}
bool libspdm_aead_aes_gcm_decrypt_prealloc(void *context, const uint8_t *key, size_t key_size,
                                           const uint8_t *iv, size_t iv_size, const uint8_t *a_data,
                                           size_t a_data_size, const uint8_t *data_in, size_t data_in_size,
                                           const uint8_t *tag, size_t tag_size, uint8_t *data_out,
                                           size_t *data_out_size)
{
    (void)context; (void)key; (void)key_size; (void)iv; (void)iv_size; (void)a_data; (void)a_data_size;
    (void)data_in; (void)data_in_size; (void)tag; (void)tag_size; (void)data_out; (void)data_out_size;
    KOT_REFUSED();
    return false;
}
void *libspdm_hmac_sha256_new(void) { KOT_REFUSED(); return NULL; }
void libspdm_hmac_sha256_free(void *hmac_sha256_ctx) { (void)hmac_sha256_ctx; KOT_REFUSED(); }
bool libspdm_hmac_sha256_set_key(void *hmac_sha256_ctx, const uint8_t *key, size_t key_size)
{
    (void)hmac_sha256_ctx; (void)key; (void)key_size;
    KOT_REFUSED();
    return false;
}
bool libspdm_hmac_sha256_update(void *hmac_sha256_ctx, const void *data, size_t data_size)
{
    (void)hmac_sha256_ctx; (void)data; (void)data_size;
    KOT_REFUSED();
    return false;
}
bool libspdm_hmac_sha256_final(void *hmac_sha256_ctx, uint8_t *hmac_value)
{
    (void)hmac_sha256_ctx; (void)hmac_value;
    KOT_REFUSED();
    return false;
}

/* ======================================================================== */
/* nv.c globals and Linux glue referenced by kernel-open/nvidia/nv-pci.c     */
/* (types as in nv-linux.h / nv-proto.h / nv-reg.h; not RM-facing)           */
/* ======================================================================== */
typedef struct nv_linux_state_s nv_linux_state_t;

nv_cap_t *nvidia_caps_root = NULL;
NvU32 NVreg_RegisterPCIDriver = 1;            /* nv-reg.h default */
NvU32 NVreg_GrdmaPciTopoCheckOverride = 0;    /* nv-reg.h default */
NvU32 num_nv_devices = 0;
NvU32 num_probed_nv_devices = 0;
NvBool nv_ats_supported = NV_FALSE;
const char *nv_device_name = "nvidia-kot-macos";
nv_linux_state_t *nv_linux_devices = NULL;
struct semaphore nv_linux_devices_lock;

nv_linux_state_t *find_uuid(const NvU8 *uuid) { (void)uuid; return NULL; }
NvBool nv_is_uuid_in_gpu_exclusion_list(const char *uuid) { (void)uuid; return NV_FALSE; }
int nv_linux_add_device_locked(nv_linux_state_t *nvl) { (void)nvl; KOT_REFUSED(); return -19; /* -ENODEV */ }
void nv_linux_remove_device_locked(nv_linux_state_t *nvl) { (void)nvl; KOT_REFUSED(); }
int nv_linux_init_open_q(nv_linux_state_t *nvl) { (void)nvl; KOT_REFUSED(); return -19; }
void nv_linux_stop_open_q(nv_linux_state_t *nvl) { (void)nvl; KOT_REFUSED(); }
NvBool nv_lock_init_locks(nvidia_stack_t *sp, nv_state_t *nv) { (void)sp; (void)nv; KOT_REFUSED(); return NV_FALSE; }
void nv_lock_destroy_locks(nvidia_stack_t *sp, nv_state_t *nv) { (void)sp; (void)nv; KOT_REFUSED(); }
void nv_dev_free_stacks(nv_linux_state_t *nvl) { (void)nvl; KOT_REFUSED(); }
NV_STATUS nv_parse_per_device_option_string(nvidia_stack_t *sp) { (void)sp; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
void nv_shutdown_adapter(nvidia_stack_t *sp, nv_state_t *nv, nv_linux_state_t *nvl) { (void)sp; (void)nv; (void)nvl; KOT_REFUSED(); }
void nv_acpi_unregister_notifier(nv_linux_state_t *nvl) { (void)nvl; KOT_REFUSED(); }
NvBool nv_acpi_power_resource_method_present(struct pci_dev *dev) { (void)dev; KOT_REFUSED(); return NV_FALSE; }

/* ======================================================================== */
/* nv-mmap.c / nv-i2c.c entry points                                         */
/* ======================================================================== */
/* No user mappings of device memory exist on this path, so revocation is
 * trivially complete; the mmap lock is still a real lock. */
static void *g_mmap_lock;
static NvBool g_safe_to_mmap = NV_TRUE;
void NV_API_CALL nv_acquire_mmap_lock(nv_state_t *nv)
{
    (void)nv;
    if (g_mmap_lock == NULL && os_alloc_mutex(&g_mmap_lock) != NV_OK)
        return;
    os_acquire_mutex(g_mmap_lock);
}
void NV_API_CALL nv_release_mmap_lock(nv_state_t *nv)
{
    (void)nv;
    if (g_mmap_lock != NULL)
        os_release_mutex(g_mmap_lock);
}
NvBool NV_API_CALL nv_get_all_mappings_revoked_locked(nv_state_t *nv) { (void)nv; return NV_TRUE; }
void NV_API_CALL nv_set_safe_to_mmap_locked(nv_state_t *nv, NvBool safe) { (void)nv; g_safe_to_mmap = safe; }
NV_STATUS NV_API_CALL nv_revoke_gpu_mappings(nv_state_t *nv) { (void)nv; return NV_OK; }

void *NV_API_CALL nv_i2c_add_adapter(nv_state_t *nv, NvU32 port) { (void)nv; (void)port; KOT_REFUSED(); return NULL; }
void NV_API_CALL nv_i2c_del_adapter(nv_state_t *nv, void *data) { (void)nv; (void)data; KOT_REFUSED(); }
