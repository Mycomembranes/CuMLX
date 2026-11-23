/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_device.c -- see kot_device.h. Plain C.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_driverkit_kpi.h>
#include <shim_mutation_admission.h>

#include "kot_device.h"
#include "kot_lifecycle.h"
#include "kot_write_policy.h"

#define KOT_CFG_SIZE 0x1000u

static pthread_mutex_t g_dev_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_state;                 /* 0 not run, 1 accepted, -1 refused */
static char g_refusal[192];
static CudaDriverKitDeviceInfo g_info;
static int g_caps_done;
static struct kot_pci_caps g_caps;
static int g_irq_done;
static struct kot_irq_config g_irq = { 0, -1, -1, -1 };

static int kot_parse_registry_id(const char *s, uint64_t *out)
{
    if (s == NULL || strlen(s) != 16)
        return -1;
    uint64_t v = 0;
    for (int i = 0; i < 16; i++) {
        char c = s[i];
        int d = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (d < 0)
            return -1;
        v = (v << 4) | (uint64_t)d;
    }
    if (v == 0)
        return -1;
    *out = v;
    return 0;
}

static void kot_device_irq_report(void);

static int kot_device_gate_locked(void)
{
    if (g_state != 0)
        return g_state;
    uint64_t want = 0;
    const char *env = getenv("KOT_REGISTRY_ENTRY_ID");
    uint32_t count = 0;
    CudaDriverKitDeviceInfo info;
    memset(&info, 0, sizeof(info));
    if (kot_parse_registry_id(env, &want) != 0) {
        snprintf(g_refusal, sizeof(g_refusal),
                 "KOT_REGISTRY_ENTRY_ID must name the card as exactly 16 hex digits (got %s)", env ? env : "nothing");
    } else if (cudaDriverKitKpiDeviceCount(0, &count) != 0 || count != 1) {
        snprintf(g_refusal, sizeof(g_refusal),
                 "the KPI reports %u NVIDIA devices; exactly one is required (KPI calls carry no device handle)",
                 (unsigned)count);
    } else if (cudaDriverKitKpiResolveDevice(CUDK_BACKEND_NVIDIA_DRIVERKIT, -1, 0, &info) != 0) {
        snprintf(g_refusal, sizeof(g_refusal), "cudaDriverKitKpiResolveDevice found no NVIDIA device");
    } else if (info.pci_vendor != 0x10de) {
        snprintf(g_refusal, sizeof(g_refusal), "resolved device vendor 0x%04x is not NVIDIA", info.pci_vendor);
    } else if (info.registry_entry_id != want) {
        snprintf(g_refusal, sizeof(g_refusal),
                 "KPI device registry entry id 0x%016llx != KOT_REGISTRY_ENTRY_ID 0x%016llx",
                 (unsigned long long)info.registry_entry_id, (unsigned long long)want);
    } else {
        g_info = info;
        g_state = 1;
        g_refusal[0] = 0;
        kot_lifecycle_note_device_resolved();
        fprintf(stderr, "kot-device: accepted 10de:%04x registry entry id 0x%016llx (BAR0 0x%llx bytes, BAR1 0x%llx bytes)\n",
                info.pci_device, (unsigned long long)info.registry_entry_id,
                (unsigned long long)info.bar0_length, (unsigned long long)info.bar1_length);
        return g_state;
    }
    g_state = -1;
    fprintf(stderr, "kot-device: REFUSED, no device access: %s\n", g_refusal);
    return g_state;
}

int kot_device_ready(void)
{
    pthread_mutex_lock(&g_dev_lock);
    int st = kot_device_gate_locked();
    pthread_mutex_unlock(&g_dev_lock);
    if (st == 1)
        kot_device_irq_report();
    return st == 1;
}

const char *kot_device_refusal(void) { return g_refusal; }
uint64_t kot_device_registry_entry_id(void) { return kot_device_ready() ? g_info.registry_entry_id : 0; }
uint16_t kot_device_pci_vendor(void) { return kot_device_ready() ? g_info.pci_vendor : 0; }
uint16_t kot_device_pci_device(void) { return kot_device_ready() ? g_info.pci_device : 0; }

int kot_device_bar(unsigned slot, uint64_t *phys_base, uint64_t *size)
{
    if (!kot_device_ready())
        return -1;
    uint64_t base = 0, len = 0;
    if (slot == 0) {
        base = KOT_DEV_BAR0_PHYS;
        len = g_info.bar0_length;
    } else if (slot == 1) {
        base = KOT_DEV_BAR1_PHYS;
        len = g_info.bar1_length;
    } else {
        return -1;
    }
    if (len == 0)
        return -1;
    if (phys_base)
        *phys_base = base;
    if (size)
        *size = len;
    return 0;
}

int kot_device_bar_for_phys(uint64_t phys, uint64_t len, unsigned *slot, uint64_t *offset)
{
    for (unsigned s = 0; s < KOT_DEV_NUM_BARS; s++) {
        uint64_t base, size;
        if (kot_device_bar(s, &base, &size) != 0)
            continue;
        if (phys < base || phys - base >= size || len > size - (phys - base) || len == 0)
            continue;
        if (slot)
            *slot = s;
        if (offset)
            *offset = phys - base;
        return 0;
    }
    return -1;
}

/* ---- config space ---------------------------------------------------------- */
int kot_cfg_read32(uint32_t aligned_offset, uint32_t *out)
{
    *out = 0xffffffffu;
    if ((aligned_offset & 3u) != 0 || aligned_offset >= KOT_CFG_SIZE)
        return KOT_DEV_BAD_ARG;
    if (!kot_device_ready())
        return KOT_DEV_REFUSED;
    uint32_t v = 0xffffffffu;
    if (cudaDriverKitKpiPciReadConfig(aligned_offset, 4, &v) != 0)
        return KOT_DEV_IO_ERROR;
    *out = v;
    return KOT_DEV_OK;
}

int kot_cfg_read(uint32_t offset, uint32_t width, uint32_t *out)
{
    uint32_t ones = width == 4 ? 0xffffffffu : ((1u << (width * 8)) - 1u);
    *out = ones;
    if ((width != 1 && width != 2 && width != 4) || (offset & (width - 1)) != 0 || offset >= KOT_CFG_SIZE)
        return KOT_DEV_BAD_ARG;
    uint32_t dw;
    int rc = kot_cfg_read32(offset & ~3u, &dw);
    if (rc != KOT_DEV_OK)
        return rc;
    *out = (dw >> ((offset & 3u) * 8)) & ones;
    return KOT_DEV_OK;
}

int kot_cfg_write(uint32_t offset, uint32_t width, uint32_t value, const char *tag)
{
    if ((width != 1 && width != 2 && width != 4) || (offset & (width - 1)) != 0 || offset >= KOT_CFG_SIZE)
        return KOT_DEV_BAD_ARG;
    if (!kot_device_ready())
        return KOT_DEV_REFUSED;
    uint32_t aligned = offset & ~3u;
    const char *rule = "";
    enum kot_wp_verdict v = kot_wp_check_cfg(aligned, &rule);
    if (v == KOT_WP_DENY) {
        kot_wp_note_denied(KOT_WP_CFG, 0, offset, value, rule, tag);
        return KOT_DEV_DENIED;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_PCI_WRITE_CONFIG) != 0)
        return KOT_DEV_NOT_ADMITTED;
    uint32_t dw = value;
    if (width != 4) {
        uint32_t shift = (offset & 3u) * 8;
        uint32_t mask = ((1u << (width * 8)) - 1u) << shift;
        uint32_t old;
        if (kot_cfg_read32(aligned, &old) != KOT_DEV_OK)
            return KOT_DEV_IO_ERROR;   /* never write back after a failed read */
        dw = (old & ~mask) | ((value << shift) & mask);
    }
    if (v == KOT_WP_ALLOW_LOG)
        kot_wp_note_logged(KOT_WP_CFG, 0, aligned, dw, rule, tag);
    int rc = cudaDriverKitKpiPciWriteConfig(aligned, 4, dw);
    kot_write_log_record(KOT_WP_CFG, 0, aligned, dw, 4, tag, __builtin_return_address(0), rc);
    return rc == 0 ? KOT_DEV_OK : KOT_DEV_IO_ERROR;
}

/* ---- capability walk ------------------------------------------------------- */
static void kot_caps_walk(struct kot_pci_caps *c)
{
    memset(c, 0, sizeof(*c));
    uint32_t dw;
    if (kot_cfg_read32(0x04, &dw) != KOT_DEV_OK) {
        snprintf(c->why, sizeof(c->why), "command/status read failed");
        return;
    }
    if (((dw >> 16) & 0x10u) == 0) {
        snprintf(c->why, sizeof(c->why), "status reports no capability list (0x%08x)", dw);
        return;
    }
    if (kot_cfg_read32(0x34, &dw) != KOT_DEV_OK) {
        snprintf(c->why, sizeof(c->why), "capability pointer read failed");
        return;
    }
    uint32_t ptr = dw & 0xfcu;
    for (int hops = 0; ptr != 0; hops++) {
        if (hops >= 48 || ptr < 0x40 || ptr > 0xfc) {
            snprintf(c->why, sizeof(c->why), "capability list loops or leaves 0x40..0xfc (at 0x%02x)", ptr);
            return;
        }
        if (kot_cfg_read32(ptr, &dw) != KOT_DEV_OK) {
            snprintf(c->why, sizeof(c->why), "capability read at 0x%02x failed", ptr);
            return;
        }
        uint32_t id = dw & 0xffu;
        if (id == 0x01 && !c->pm) c->pm = ptr;
        if (id == 0x05 && !c->msi) c->msi = ptr;
        if (id == 0x11 && !c->msix) c->msix = ptr;
        if (id == 0x10 && !c->pcie) c->pcie = ptr;
        ptr = (dw >> 8) & 0xfcu;
    }
    if (!c->pcie) {
        snprintf(c->why, sizeof(c->why), "no PCI Express capability found");
        return;
    }
    c->ok = 1;
    snprintf(c->why, sizeof(c->why), "pm 0x%02x msi 0x%02x msix 0x%02x pcie 0x%02x", c->pm, c->msi, c->msix, c->pcie);
}

const struct kot_pci_caps *kot_device_caps(void)
{
    pthread_mutex_lock(&g_dev_lock);
    int done = g_caps_done;
    pthread_mutex_unlock(&g_dev_lock);
    if (done)
        return &g_caps;
    struct kot_pci_caps c;
    kot_caps_walk(&c);
    pthread_mutex_lock(&g_dev_lock);
    if (!g_caps_done) {
        g_caps = c;
        g_caps_done = 1;
        fprintf(stderr, "kot-device: capability walk %s: %s\n", c.ok ? "ok" : "FAILED (all config writes denied)", c.why);
    }
    pthread_mutex_unlock(&g_dev_lock);
    return &g_caps;
}

static void kot_device_irq_report(void)
{
    pthread_mutex_lock(&g_dev_lock);
    int done = g_irq_done;
    g_irq_done = 1;
    pthread_mutex_unlock(&g_dev_lock);
    if (done)
        return;
    struct kot_irq_config q = { 1, -1, -1, -1 };
    uint32_t dw;
    if (kot_cfg_read32(0x04, &dw) == KOT_DEV_OK)
        q.intx_disabled = (dw >> 10) & 1u;
    const struct kot_pci_caps *c = kot_device_caps();
    if (c->ok && c->msi && kot_cfg_read32(c->msi, &dw) == KOT_DEV_OK)
        q.msi_enabled = (dw >> 16) & 1u;
    else if (c->ok && !c->msi)
        q.msi_enabled = 0;
    if (c->ok && c->msix && kot_cfg_read32(c->msix, &dw) == KOT_DEV_OK)
        q.msix_enabled = (dw >> 31) & 1u;
    else if (c->ok && !c->msix)
        q.msix_enabled = 0;
    pthread_mutex_lock(&g_dev_lock);
    g_irq = q;
    pthread_mutex_unlock(&g_dev_lock);
    fprintf(stderr, "kot-device: interrupts (read-only): INTx disabled %d, MSI enabled %d, MSI-X enabled %d "
                    "(-1 = unknown); this layer services interrupts only by polling (macos_os_priv.h)\n",
            q.intx_disabled, q.msi_enabled, q.msix_enabled);
}

const struct kot_irq_config *kot_device_irq_config(void)
{
    (void)kot_device_ready();
    return &g_irq;
}

void kot_device_reset_for_tests(void)
{
    pthread_mutex_lock(&g_dev_lock);
    g_state = 0;
    g_refusal[0] = 0;
    memset(&g_info, 0, sizeof(g_info));
    g_caps_done = 0;
    memset(&g_caps, 0, sizeof(g_caps));
    g_irq_done = 0;
    g_irq.read = 0;
    g_irq.intx_disabled = g_irq.msi_enabled = g_irq.msix_enabled = -1;
    pthread_mutex_unlock(&g_dev_lock);
}
