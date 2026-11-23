/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/os_pci_macos.c -- macOS replacement for
 * kernel-open/nvidia/os-pci.c (OGK 570.190). One device only: the card the
 * identity gate accepted (kot_device.h: KOT_REGISTRY_ENTRY_ID must equal the
 * registry entry id the KPI reports, with exactly one NVIDIA device present).
 * Config space goes through kot_cfg_read / kot_cfg_write, the single config
 * choke point (also used by shims/shim_pci.c): write policy (kot_write_policy.h),
 * shim_mutation_admit (SHIM_MUTATION_PREGATE is defined for this build), aligned
 * dword read-modify-write for byte and word writes with no write-back after a
 * failed read, and the write ring log. FLR, removal and power management are
 * refused.
 */
#include <stdio.h>
#include <string.h>

#include "nvtypes.h"
#include "nvstatus.h"
#include "os-interface.h"
#include "nv.h"

#include "macos_os_priv.h"
#include "kot_device.h"

/* The opaque handle RM gets back; it only ever names the accepted device. */
static struct kot_pci_handle {
    NvU32 magic;
    NvU32 domain;
    NvU8 bus, slot, function;
} g_gpu = { 0x6b6f7470u /* 'kotp' */, 0, 0, 0, 0 };

void kot_pci_set_gpu_location(uint32_t domain, uint8_t bus, uint8_t slot, uint8_t function)
{
    g_gpu.domain = domain;
    g_gpu.bus = bus;
    g_gpu.slot = slot;
    g_gpu.function = function;
}

static int kot_is_gpu(void *handle) { return handle == (void *)&g_gpu; }

void *NV_API_CALL os_pci_init_handle(NvU32 domain, NvU8 bus, NvU8 slot, NvU8 function,
                                     NvU16 *vendor, NvU16 *device)
{
    if (domain != g_gpu.domain || bus != g_gpu.bus || slot != g_gpu.slot || function != g_gpu.function)
        return NULL;   /* no other PCI function is reachable from here */
    if (!kot_device_ready())
        return NULL;   /* identity gate refused (reason on stderr) */
    if (vendor) *vendor = kot_device_pci_vendor();
    if (device) *device = kot_device_pci_device();
    return &g_gpu;
}

static NV_STATUS kot_status(int rc)
{
    switch (rc) {
    case KOT_DEV_OK:           return NV_OK;
    case KOT_DEV_DENIED:       return NV_ERR_NOT_SUPPORTED;
    case KOT_DEV_BAD_ARG:      return NV_ERR_NOT_SUPPORTED;
    case KOT_DEV_NOT_ADMITTED: return NV_ERR_INSUFFICIENT_PERMISSIONS;
    case KOT_DEV_REFUSED:      return NV_ERR_INVALID_ARGUMENT;
    default:                   return NV_ERR_OPERATING_SYSTEM;
    }
}

static NV_STATUS kot_cfg_read_h(void *handle, NvU32 offset, NvU32 width, NvU32 *out)
{
    NvU32 ones = (width == 4) ? 0xffffffffu : ((1u << (width * 8)) - 1u);
    if (!kot_is_gpu(handle)) {
        *out = ones;
        return NV_ERR_INVALID_ARGUMENT;
    }
    uint32_t v = ones;
    NV_STATUS st = kot_status(kot_cfg_read(offset, width, &v));
    *out = v;
    return st;
}

static NV_STATUS kot_cfg_write_h(void *handle, NvU32 offset, NvU32 width, NvU32 value, const char *tag)
{
    if (!kot_is_gpu(handle))
        return NV_ERR_INVALID_ARGUMENT;
    return kot_status(kot_cfg_write(offset, width, value, tag));
}

NV_STATUS NV_API_CALL os_pci_read_byte(void *handle, NvU32 offset, NvU8 *pReturnValue)
{
    NvU32 v;
    NV_STATUS st = kot_cfg_read_h(handle, offset, 1, &v);
    *pReturnValue = (NvU8)v;
    return st;
}

NV_STATUS NV_API_CALL os_pci_read_word(void *handle, NvU32 offset, NvU16 *pReturnValue)
{
    NvU32 v;
    NV_STATUS st = kot_cfg_read_h(handle, offset, 2, &v);
    *pReturnValue = (NvU16)v;
    return st;
}

NV_STATUS NV_API_CALL os_pci_read_dword(void *handle, NvU32 offset, NvU32 *pReturnValue)
{
    return kot_cfg_read_h(handle, offset, 4, pReturnValue);
}

NV_STATUS NV_API_CALL os_pci_write_byte(void *handle, NvU32 offset, NvU8 value)
{
    return kot_cfg_write_h(handle, offset, 1, value, "os_pci_write_byte");
}

NV_STATUS NV_API_CALL os_pci_write_word(void *handle, NvU32 offset, NvU16 value)
{
    return kot_cfg_write_h(handle, offset, 2, value, "os_pci_write_word");
}

NV_STATUS NV_API_CALL os_pci_write_dword(void *handle, NvU32 offset, NvU32 value)
{
    return kot_cfg_write_h(handle, offset, 4, value, "os_pci_write_dword");
}

/* Never: removal, FLR (operator-gated silicon actions). */
NvBool NV_API_CALL os_pci_remove_supported(void) { return NV_FALSE; }
void NV_API_CALL os_pci_remove(void *handle) { (void)handle; KOT_REFUSED(); }
void NV_API_CALL os_pci_trigger_flr(void *handle) { (void)handle; KOT_REFUSED(); }
