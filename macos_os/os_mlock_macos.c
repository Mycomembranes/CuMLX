/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/os_mlock_macos.c -- macOS replacement for
 * kernel-open/nvidia/os-mlock.c (OGK 570.190). Pinning user pages for device
 * DMA needs a kernel; this userspace layer refuses all three entry points.
 * Device-visible memory comes only from nv_alloc_pages() (KPI DMA).
 */
#include "nvtypes.h"
#include "nvstatus.h"
#include "os-interface.h"

#include "macos_os_priv.h"

NV_STATUS NV_API_CALL os_lookup_user_io_memory(void *address, NvU64 page_count, NvU64 **pte_array)
{
    (void)address; (void)page_count;
    KOT_REFUSED();
    if (pte_array) *pte_array = NULL;
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS NV_API_CALL os_lock_user_pages(void *address, NvU64 page_count, void **page_array, NvU32 flags)
{
    (void)address; (void)page_count; (void)flags;
    KOT_REFUSED();
    if (page_array) *page_array = NULL;
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS NV_API_CALL os_unlock_user_pages(NvU64 page_count, void *page_array, NvU32 flags)
{
    (void)page_count; (void)page_array; (void)flags;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
