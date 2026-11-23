/*
 * shims/shim_io.c -- BAR MMIO for the nvidia-ko translator.
 *
 * Two variants:
 *   KOT_SAFE_SHIMS (the RM closure link, shims/macos_os, and the host tests):
 *     BAR tokens with a registry (shims/macos_os/kot_tokens.h), the identity
 *     gate (kot_device.h), the local write policy and the write ring log
 *     (kot_write_policy.h). This is the only variant meant to sit in front of
 *     a card.
 *   otherwise (the default 570.190 / 595.71.05 translator build): the legacy
 *     token scheme, kept byte-identical for those objects. It has no bounds,
 *     identity or write-policy checks and is NOT fit to be put in front of a
 *     card.
 */
#if defined(KOT_SAFE_SHIMS)
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_driverkit_kpi.h>
#include <linux/io.h>
#include <shim_mutation_admission.h>

#include "macos_os/kot_device.h"
#include "macos_os/kot_tokens.h"
#include "macos_os/kot_write_policy.h"

struct kot_map {
    uint32_t state;   /* 0 never used, 1 live, 2 unmapped */
    uint32_t slot;
    uint64_t off_base, size;
};
static pthread_mutex_t g_tok_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kot_map g_maps[KOT_TOKEN_IDX_MAX + 1];
static uint32_t g_next_idx = 1;   /* index 0 is never issued; indices are never reused */
static struct kot_token_stats g_ts;
static int g_once[8];

enum { KOT_ONCE_BAD, KOT_ONCE_STALE, KOT_ONCE_RANGE, KOT_ONCE_ALIGN, KOT_ONCE_REFUSED, KOT_ONCE_READ, KOT_ONCE_UNMAP, KOT_ONCE_MAP };

static void kot_tok_note(int kind, uint64_t *ctr, const char *what, const volatile void *addr, const char *tag)
{
    pthread_mutex_lock(&g_tok_lock);
    (*ctr)++;
    int first = !g_once[kind];
    g_once[kind] = 1;
    pthread_mutex_unlock(&g_tok_lock);
    if (first)
        fprintf(stderr, "kot-io: %s: %s %p refused (all-ones / no write; reported once per kind, counted)\n",
                tag, what, (const void *)addr);
}

void *kot_token_map(unsigned slot, uint64_t bar_offset, uint64_t size)
{
    uint64_t base, bar_size;
    if (kot_device_bar(slot, &base, &bar_size) != 0 || size == 0 || bar_offset >= bar_size ||
        size > bar_size - bar_offset || bar_offset > KOT_TOKEN_OFF_MASK) {
        kot_tok_note(KOT_ONCE_MAP, &g_ts.map_refused, "mapping outside the BARs of the accepted device", 0, "kot_token_map");
        return NULL;
    }
    pthread_mutex_lock(&g_tok_lock);
    if (g_next_idx > KOT_TOKEN_IDX_MAX) {
        pthread_mutex_unlock(&g_tok_lock);
        kot_tok_note(KOT_ONCE_MAP, &g_ts.map_refused, "token registry exhausted", 0, "kot_token_map");
        return NULL;
    }
    uint32_t idx = g_next_idx++;
    g_maps[idx].state = 1;
    g_maps[idx].slot = slot;
    g_maps[idx].off_base = bar_offset;
    g_maps[idx].size = size;
    g_ts.mapped++;
    pthread_mutex_unlock(&g_tok_lock);
    return (void *)(uintptr_t)(KOT_TOKEN_BASE | ((uint64_t)idx << KOT_TOKEN_IDX_SHIFT) | bar_offset);
}

int kot_token_unmap(const volatile void *token)
{
    uint64_t t = (uint64_t)(uintptr_t)token;
    uint64_t idx = (t >> KOT_TOKEN_IDX_SHIFT) & KOT_TOKEN_IDX_MAX;
    int rc = -1;
    pthread_mutex_lock(&g_tok_lock);
    if ((t & ~((1ull << 52) - 1)) == KOT_TOKEN_BASE && idx != 0 && g_maps[idx].state == 1 &&
        (t & KOT_TOKEN_OFF_MASK) == g_maps[idx].off_base) {
        g_maps[idx].state = 2;
        g_ts.unmapped++;
        rc = 0;
    }
    pthread_mutex_unlock(&g_tok_lock);
    if (rc != 0)
        kot_tok_note(KOT_ONCE_UNMAP, &g_ts.unmap_unknown, "unmap of an unknown or already unmapped token", token, "iounmap");
    return rc;
}

/* Validate addr for a width-byte access; KPI bar index and BAR offset out. */
static int kot_tok_decode(const volatile void *addr, uint32_t width, uint32_t *bar, uint32_t *off, const char *tag)
{
    uint64_t a = (uint64_t)(uintptr_t)addr;
    if ((a & ~((1ull << 52) - 1)) != KOT_TOKEN_BASE) {
        kot_tok_note(KOT_ONCE_BAD, &g_ts.bad_token, "address that is not a BAR token", addr, tag);
        return -1;
    }
    uint64_t idx = (a >> KOT_TOKEN_IDX_SHIFT) & KOT_TOKEN_IDX_MAX;
    uint64_t o = a & KOT_TOKEN_OFF_MASK;
    if ((o & (width - 1)) != 0) {
        kot_tok_note(KOT_ONCE_ALIGN, &g_ts.misaligned, "misaligned access through token", addr, tag);
        return -1;
    }
    pthread_mutex_lock(&g_tok_lock);
    struct kot_map m = g_maps[idx];
    pthread_mutex_unlock(&g_tok_lock);
    if (idx == 0 || m.state != 1) {
        kot_tok_note(KOT_ONCE_STALE, &g_ts.stale_token, m.state == 2 ? "stale token (unmapped)" : "unknown token", addr, tag);
        return -1;
    }
    uint64_t base, bar_size;
    if (!kot_device_ready() || kot_device_bar(m.slot, &base, &bar_size) != 0) {
        kot_tok_note(KOT_ONCE_REFUSED, &g_ts.device_refused, "access with the device not accepted", addr, tag);
        return -1;
    }
    if (o < m.off_base || o - m.off_base >= m.size || width > m.size - (o - m.off_base) ||
        o >= bar_size || width > bar_size - o || o > 0xffffffffull) {
        kot_tok_note(KOT_ONCE_RANGE, &g_ts.out_of_range, "access outside the mapping or the BAR", addr, tag);
        return -1;
    }
    *bar = m.slot;
    *off = (uint32_t)o;
    return 0;
}

void kot_token_stats(struct kot_token_stats *out)
{
    pthread_mutex_lock(&g_tok_lock);
    *out = g_ts;
    pthread_mutex_unlock(&g_tok_lock);
}

void kot_token_reset_for_tests(void)
{
    pthread_mutex_lock(&g_tok_lock);
    for (uint32_t i = 0; i <= KOT_TOKEN_IDX_MAX; i++)
        if (g_maps[i].state == 1)
            g_maps[i].state = 2;   /* keep "never reused" across resets */
    memset(&g_ts, 0, sizeof(g_ts));
    memset(g_once, 0, sizeof(g_once));
    pthread_mutex_unlock(&g_tok_lock);
}

void __iomem *ioremap(phys_addr_t offset, unsigned long size)
{
    unsigned slot;
    uint64_t bar_off;
    if (kot_device_bar_for_phys((uint64_t)offset, (uint64_t)size, &slot, &bar_off) != 0) {
        kot_tok_note(KOT_ONCE_MAP, &g_ts.map_refused, "ioremap of a range outside the device BARs",
                     (const void *)(uintptr_t)offset, "ioremap");
        return NULL;
    }
    return (void __iomem *)kot_token_map(slot, bar_off, (uint64_t)size);
}

void iounmap(volatile void __iomem *addr)
{
    if (addr != NULL)
        (void)kot_token_unmap(addr);
}

static int kot_mmio_read32(uint32_t bar, uint32_t aligned, uint32_t *v, const volatile void *addr, const char *tag)
{
    *v = 0xffffffffU;
    uint32_t tmp = 0xffffffffU;
    if (cudaDriverKitKpiReadReg32(bar, aligned, &tmp) != 0) {
        kot_tok_note(KOT_ONCE_READ, &g_ts.read_failed, "KPI register read failed", addr, tag);
        return -1;
    }
    *v = tmp;
    return 0;
}

u8 shim_readb(const volatile void __iomem *addr)
{
    uint32_t bar, off, v;
    if (kot_tok_decode(addr, 1, &bar, &off, "shim_readb") != 0 ||
        kot_mmio_read32(bar, off & ~3U, &v, addr, "shim_readb") != 0)
        return 0xffU;
    return (u8)(v >> ((off & 3U) * 8));
}

u16 shim_readw(const volatile void __iomem *addr)
{
    uint32_t bar, off, v;
    if (kot_tok_decode(addr, 2, &bar, &off, "shim_readw") != 0 ||
        kot_mmio_read32(bar, off & ~3U, &v, addr, "shim_readw") != 0)
        return 0xffffU;
    return (u16)(v >> ((off & 2U) * 8));
}

u32 shim_readl(const volatile void __iomem *addr)
{
    uint32_t bar, off, v;
    if (kot_tok_decode(addr, 4, &bar, &off, "shim_readl") != 0 ||
        kot_mmio_read32(bar, off, &v, addr, "shim_readl") != 0)
        return 0xffffffffU;
    return v;
}

u64 shim_readq(const volatile void __iomem *addr)
{
    uint32_t lo = shim_readl(addr);
    uint32_t hi = shim_readl((const volatile void __iomem *)((uintptr_t)addr + 4));
    return ((uint64_t)hi << 32) | lo;
}

/* One dword write after policy + admission; mask != ~0 means a byte/word RMW. */
static void kot_mmio_write(volatile void __iomem *addr, uint32_t width, uint32_t v, const char *tag, const void *caller)
{
    uint32_t bar, off;
    if (kot_tok_decode(addr, width, &bar, &off, tag) != 0)
        return;
    uint32_t aligned = off & ~3U;
    const char *rule = "";
    enum kot_wp_verdict verdict = kot_wp_check_mmio(bar, aligned, &rule);
    if (verdict == KOT_WP_DENY) {
        kot_wp_note_denied(KOT_WP_MMIO, bar, aligned, v, rule, tag);
        return;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_MMIO_WRITE_REG32) != 0)
        return;
    uint32_t dw = v;
    if (width != 4) {
        uint32_t old;
        if (kot_mmio_read32(bar, aligned, &old, addr, tag) != 0)
            return;   /* never write back after a failed read */
        uint32_t shift = (off & 3U) * 8;
        uint32_t mask = (width == 1 ? 0xffU : 0xffffU) << shift;
        dw = (old & ~mask) | ((v << shift) & mask);
    }
    if (verdict == KOT_WP_ALLOW_LOG)
        kot_wp_note_logged(KOT_WP_MMIO, bar, aligned, dw, rule, tag);
    int rc = cudaDriverKitKpiWriteReg32(bar, aligned, dw);
    kot_write_log_record(KOT_WP_MMIO, bar, aligned, dw, width, tag, caller, rc);
}

void shim_writeb(u8 v, volatile void __iomem *addr)
{
    kot_mmio_write(addr, 1, v, "shim_writeb", __builtin_return_address(0));
}

void shim_writew(u16 v, volatile void __iomem *addr)
{
    kot_mmio_write(addr, 2, v, "shim_writew", __builtin_return_address(0));
}

void shim_writel(u32 v, volatile void __iomem *addr)
{
    kot_mmio_write(addr, 4, v, "shim_writel", __builtin_return_address(0));
}

void shim_writeq(u64 v, volatile void __iomem *addr)
{
    kot_mmio_write(addr, 4, (u32)v, "shim_writeq", __builtin_return_address(0));
    kot_mmio_write((volatile void __iomem *)((uintptr_t)addr + 4), 4, (u32)(v >> 32), "shim_writeq",
                   __builtin_return_address(0));
}

#else /* !KOT_SAFE_SHIMS: legacy default-build variant, unchanged */
#include <stdint.h>

#include <cuda_driverkit_kpi.h>
#include <linux/io.h>
#include <shim_mutation_admission.h>

static uint32_t shim_bar_from_token(const volatile void __iomem *addr, uint32_t *offset)
{
    uintptr_t raw = (uintptr_t)addr;
    if (raw <= 2) {
        *offset = 0;
        return raw == 2 ? 1 : 0;
    }
    *offset = (uint32_t)(raw & 0x0fffffffU);
    return (uint32_t)((raw >> 60) & 0xfU);
}

void __iomem *ioremap(phys_addr_t offset, unsigned long size)
{
    (void)size;
    return (void __iomem *)(uintptr_t)offset;
}

void iounmap(volatile void __iomem *addr) { (void)addr; }

u8 shim_readb(const volatile void __iomem *addr)
{
    uint32_t off = 0;
    uint32_t bar = shim_bar_from_token(addr, &off);
    uint32_t value = 0;
    (void)cudaDriverKitKpiReadReg32(bar, off & ~3U, &value);
    return (u8)(value >> ((off & 3U) * 8));
}

u16 shim_readw(const volatile void __iomem *addr)
{
    uint32_t off = 0;
    uint32_t bar = shim_bar_from_token(addr, &off);
    uint32_t value = 0;
    (void)cudaDriverKitKpiReadReg32(bar, off & ~3U, &value);
    return (u16)(value >> ((off & 2U) * 8));
}

u32 shim_readl(const volatile void __iomem *addr)
{
    uint32_t off = 0;
    uint32_t bar = shim_bar_from_token(addr, &off);
    uint32_t value = 0xffffffffU;
    (void)cudaDriverKitKpiReadReg32(bar, off, &value);
    return value;
}

u64 shim_readq(const volatile void __iomem *addr)
{
    uint32_t lo = shim_readl(addr);
    uint32_t hi = shim_readl((const volatile void __iomem *)((uintptr_t)addr + 4));
    return ((uint64_t)hi << 32) | lo;
}

void shim_writeb(u8 v, volatile void __iomem *addr)
{
    if (shim_mutation_admit(CUDK_MUTATION_OP_MMIO_WRITE_REG32) != 0) {
        return;
    }
    uint32_t off = 0;
    uint32_t bar = shim_bar_from_token(addr, &off);
    uint32_t aligned = off & ~3U;
    uint32_t value = 0;
    (void)cudaDriverKitKpiReadReg32(bar, aligned, &value);
    uint32_t shift = (off & 3U) * 8;
    value = (value & ~(0xffU << shift)) | ((uint32_t)v << shift);
    (void)cudaDriverKitKpiWriteReg32(bar, aligned, value);
}

void shim_writew(u16 v, volatile void __iomem *addr)
{
    if (shim_mutation_admit(CUDK_MUTATION_OP_MMIO_WRITE_REG32) != 0) {
        return;
    }
    uint32_t off = 0;
    uint32_t bar = shim_bar_from_token(addr, &off);
    uint32_t aligned = off & ~3U;
    uint32_t value = 0;
    (void)cudaDriverKitKpiReadReg32(bar, aligned, &value);
    uint32_t shift = (off & 2U) * 8;
    value = (value & ~(0xffffU << shift)) | ((uint32_t)v << shift);
    (void)cudaDriverKitKpiWriteReg32(bar, aligned, value);
}

void shim_writel(u32 v, volatile void __iomem *addr)
{
    if (shim_mutation_admit(CUDK_MUTATION_OP_MMIO_WRITE_REG32) != 0) {
        return;
    }
    uint32_t off = 0;
    uint32_t bar = shim_bar_from_token(addr, &off);
    (void)cudaDriverKitKpiWriteReg32(bar, off, v);
}

void shim_writeq(u64 v, volatile void __iomem *addr)
{
    if (shim_mutation_admit(CUDK_MUTATION_OP_MMIO_WRITE_REG32) != 0) {
        return;
    }
    shim_writel((u32)v, addr);
    shim_writel((u32)(v >> 32), (volatile void __iomem *)((uintptr_t)addr + 4));
}
#endif /* KOT_SAFE_SHIMS */
