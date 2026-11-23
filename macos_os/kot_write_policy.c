/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_write_policy.c -- see kot_write_policy.h. Plain C.
 */
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kot_device.h"
#include "kot_lifecycle.h"
#include "kot_write_policy.h"

/* ---- compiled-in MMIO table (BAR0 offsets, inclusive ranges) ------------- */
struct kot_wp_rule {
    uint32_t bar, lo, hi;
    enum kot_wp_verdict verdict;
    const char *name;
};
static const struct kot_wp_rule k_mmio_rules[] = {
    { 0, 0x000200, 0x000203, KOT_WP_DENY, "NV_PMC_ENABLE" },
    { 0, 0x000600, 0x000603, KOT_WP_DENY, "NV_PMC_DEVICE_ENABLE(0)" },
    { 0, 0x1103c0, 0x1103c3, KOT_WP_DENY, "NV_PGSP_FALCON_ENGINE (GSP falcon reset)" },
    { 0, 0x8403c0, 0x8403c3, KOT_WP_DENY, "NV_PSEC_FALCON_ENGINE (SEC2 falcon reset)" },
    { 0, 0x8f03c0, 0x8f03c3, KOT_WP_DENY, "FSP falcon ENGINE (0x8f0000+0x3c0, FSP reset)" },
    { 0, 0x088000, 0x088fff, KOT_WP_DENY, "BAR0 mirror of PCI config space (NV_PCFG)" },
    { 0, 0x1fa824, 0x1fa82b, KOT_WP_ALLOW_LOG, "NV_PFB_PRI_MMU_WPR2_ADDR_LO/HI" },
    { 0, 0x110000, 0x111fff, KOT_WP_ALLOW_LOG, "GSP falcon/RISC-V window" },
    { 0, 0x840000, 0x841fff, KOT_WP_ALLOW_LOG, "SEC2 falcon window" },
    { 0, 0x8f0000, 0x8f3fff, KOT_WP_ALLOW_LOG, "FSP falcon/queue/EMEM window" },
};

static pthread_mutex_t g_wp_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_allow_loaded;
#define KOT_WP_ALLOW_MAX 256
static struct { int cfg; uint32_t bar, off; } g_allow[KOT_WP_ALLOW_MAX];
static unsigned g_allow_n;
static uint64_t g_mmio_denied, g_cfg_denied, g_logged, g_issued;

/* once-per-address stderr (open addressing; when full, lines are printed) */
#define KOT_WP_SEEN 2048
static uint64_t g_seen[KOT_WP_SEEN];
static int kot_wp_first_time(uint64_t key)
{
    key |= 1ull << 63;   /* never 0 */
    uint64_t h = (key * 0x9e3779b97f4a7c15ull) >> 53;
    for (unsigned i = 0; i < KOT_WP_SEEN; i++) {
        uint64_t *s = &g_seen[(h + i) % KOT_WP_SEEN];
        if (*s == key)
            return 0;
        if (*s == 0) {
            *s = key;
            return 1;
        }
    }
    return 1;
}

static void kot_wp_load_allowlist_locked(void)
{
    if (g_allow_loaded)
        return;
    g_allow_loaded = 1;
    g_allow_n = 0;
    const char *p = getenv("KOT_WRITE_ALLOWLIST");
    if (p == NULL || p[0] == 0)
        return;
    FILE *f = fopen(p, "r");
    if (f == NULL) {
        fprintf(stderr, "kot-write-policy: KOT_WRITE_ALLOWLIST=%s cannot be read: no allowlist\n", p);
        return;
    }
    char line[256];
    unsigned lineno = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        lineno++;
        char *s = line;
        while (*s == ' ' || *s == '\t')
            s++;
        char *e = strpbrk(s, "#\r\n");
        if (e)
            *e = 0;
        if (*s == 0)
            continue;
        char *colon = strchr(s, ':');
        if (colon == NULL) {
            fprintf(stderr, "kot-write-policy: %s:%u: ignored (want bar:offset or cfg:offset)\n", p, lineno);
            continue;
        }
        *colon = 0;
        char *end = NULL;
        unsigned long long off = strtoull(colon + 1, &end, 0);
        while (end && (*end == ' ' || *end == '\t'))
            end++;
        if (end == colon + 1 || (end && *end != 0) || off > 0xffffffffull || (off & 3u) != 0) {
            fprintf(stderr, "kot-write-policy: %s:%u: ignored (offset must be a dword-aligned number)\n", p, lineno);
            continue;
        }
        int cfg = strcmp(s, "cfg") == 0;
        unsigned long bar = 0;
        if (!cfg) {
            const char *b = s;
            if (strncmp(b, "bar", 3) == 0)
                b += 3;
            char *be = NULL;
            bar = strtoul(b, &be, 0);
            if (be == b || *be != 0 || bar >= KOT_DEV_NUM_BARS) {
                fprintf(stderr, "kot-write-policy: %s:%u: ignored (unknown space '%s')\n", p, lineno, s);
                continue;
            }
        }
        if (g_allow_n < KOT_WP_ALLOW_MAX) {
            g_allow[g_allow_n].cfg = cfg;
            g_allow[g_allow_n].bar = (uint32_t)bar;
            g_allow[g_allow_n].off = (uint32_t)off;
            g_allow_n++;
            fprintf(stderr, "kot-write-policy: allowlisted %s:0x%llx (from %s)\n", cfg ? "cfg" : s, off, p);
        }
    }
    fclose(f);
}

static int kot_wp_allowlisted(int cfg, uint32_t bar, uint32_t off)
{
    pthread_mutex_lock(&g_wp_lock);
    kot_wp_load_allowlist_locked();
    int hit = 0;
    for (unsigned i = 0; i < g_allow_n && !hit; i++)
        hit = g_allow[i].cfg == cfg && g_allow[i].off == off && (cfg || g_allow[i].bar == bar);
    pthread_mutex_unlock(&g_wp_lock);
    return hit;
}

enum kot_wp_verdict kot_wp_check_mmio(uint32_t bar, uint32_t offset, const char **rule)
{
    const char *dummy;
    if (rule == NULL)
        rule = &dummy;
    *rule = "";
    if (kot_lifecycle_fatal_entered()) {
        *rule = "kot_fatal entered: no further device writes";
        return KOT_WP_DENY;
    }
    for (size_t i = 0; i < sizeof(k_mmio_rules) / sizeof(k_mmio_rules[0]); i++) {
        const struct kot_wp_rule *r = &k_mmio_rules[i];
        if (r->bar != bar || offset + 3u < r->lo || offset > r->hi)
            continue;
        *rule = r->name;
        if (r->verdict == KOT_WP_DENY && kot_wp_allowlisted(0, bar, offset & ~3u))
            return KOT_WP_ALLOW_LOG;
        return r->verdict;
    }
    return KOT_WP_ALLOW;
}

/* Protected config registers: [start, start+len) in bytes. */
struct kot_cfg_range { uint32_t start, len; const char *name; };

static unsigned kot_cfg_ranges(struct kot_cfg_range *out, unsigned max)
{
    unsigned n = 0;
#define KOT_ADD(s, l, nm) do { if (n < max) { out[n].start = (s); out[n].len = (l); out[n].name = (nm); n++; } } while (0)
    KOT_ADD(0x04, 2, "PCI command");
    KOT_ADD(0x10, 4, "PCI BAR0");
    KOT_ADD(0x14, 4, "PCI BAR1");
    KOT_ADD(0x18, 4, "PCI BAR2");
    KOT_ADD(0x1c, 4, "PCI BAR3");
    KOT_ADD(0x20, 4, "PCI BAR4");
    KOT_ADD(0x24, 4, "PCI BAR5");
    KOT_ADD(0x30, 4, "PCI expansion ROM BAR");
    const struct kot_pci_caps *c = kot_device_caps();
    if (c != NULL && c->ok) {
        if (c->pm)   KOT_ADD(c->pm + 4, 2, "PM capability PMCSR");
        if (c->msi)  KOT_ADD(c->msi + 2, 2, "MSI message control");
        if (c->msix) KOT_ADD(c->msix + 2, 2, "MSI-X message control");
        if (c->pcie) KOT_ADD(c->pcie + 8, 2, "PCIe Device Control");
        if (c->pcie) KOT_ADD(c->pcie + 0x10, 2, "PCIe Link Control");
    }
#undef KOT_ADD
    return n;
}

enum kot_wp_verdict kot_wp_check_cfg(uint32_t aligned_offset, const char **rule)
{
    const char *dummy;
    if (rule == NULL)
        rule = &dummy;
    *rule = "";
    if (kot_lifecycle_fatal_entered()) {
        *rule = "kot_fatal entered: no further device writes";
        return KOT_WP_DENY;
    }
    const struct kot_pci_caps *c = kot_device_caps();
    if (c == NULL || !c->ok) {
        *rule = "capability walk failed: every config write denied";
        return KOT_WP_DENY;
    }
    if (aligned_offset >= 0x100) {
        *rule = "extended config space (AER, Resizable BAR, ...)";
        return kot_wp_allowlisted(1, 0, aligned_offset) ? KOT_WP_ALLOW_LOG : KOT_WP_DENY;
    }
    struct kot_cfg_range r[16];
    unsigned n = kot_cfg_ranges(r, 16);
    int hit = 0, all_allowed = 1;
    for (unsigned i = 0; i < n; i++) {
        if (r[i].start + r[i].len <= aligned_offset || r[i].start >= aligned_offset + 4)
            continue;
        hit = 1;
        *rule = r[i].name;
        if (!kot_wp_allowlisted(1, 0, r[i].start & ~3u) && !kot_wp_allowlisted(1, 0, r[i].start))
            all_allowed = 0;
    }
    if (!hit)
        return KOT_WP_ALLOW;
    return all_allowed ? KOT_WP_ALLOW_LOG : KOT_WP_DENY;
}

void kot_wp_note_denied(enum kot_wp_space space, uint32_t bar, uint32_t offset, uint32_t value,
                        const char *rule, const char *tag)
{
    pthread_mutex_lock(&g_wp_lock);
    if (space == KOT_WP_CFG)
        g_cfg_denied++;
    else
        g_mmio_denied++;
    int first = kot_wp_first_time(((uint64_t)space << 40) | ((uint64_t)bar << 32) | offset);
    pthread_mutex_unlock(&g_wp_lock);
    if (first)
        fprintf(stderr, "kot-write-policy: DENIED %s %s0x%06x <- 0x%08x (%s; via %s)\n",
                space == KOT_WP_CFG ? "cfg" : "mmio", space == KOT_WP_CFG ? "" : (bar ? "bar1+" : "bar0+"),
                offset, value, rule ? rule : "", tag ? tag : "?");
}

void kot_wp_note_logged(enum kot_wp_space space, uint32_t bar, uint32_t offset, uint32_t value,
                        const char *rule, const char *tag)
{
    pthread_mutex_lock(&g_wp_lock);
    g_logged++;
    int first = kot_wp_first_time((1ull << 48) | ((uint64_t)space << 40) | ((uint64_t)bar << 32) | offset);
    pthread_mutex_unlock(&g_wp_lock);
    if (first)
        fprintf(stderr, "kot-write-policy: logged %s %s0x%06x <- 0x%08x (%s; via %s; first write to this address)\n",
                space == KOT_WP_CFG ? "cfg" : "mmio", space == KOT_WP_CFG ? "" : (bar ? "bar1+" : "bar0+"),
                offset, value, rule ? rule : "", tag ? tag : "?");
}

/* ---- ring log ------------------------------------------------------------ */
#define KOT_WLOG 4096
struct kot_wlog {
    uint64_t seq;
    uint32_t space, bar, offset, value, width;
    int rc;
    const char *tag;
    const void *caller;
};
static struct kot_wlog g_wlog[KOT_WLOG];
static uint64_t g_wlog_n;

void kot_write_log_record(enum kot_wp_space space, uint32_t bar, uint32_t offset, uint32_t value,
                          uint32_t width, const char *tag, const void *caller, int rc)
{
    pthread_mutex_lock(&g_wp_lock);
    struct kot_wlog *e = &g_wlog[g_wlog_n % KOT_WLOG];
    e->seq = g_wlog_n;
    e->space = (uint32_t)space;
    e->bar = bar;
    e->offset = offset;
    e->value = value;
    e->width = width;
    e->rc = rc;
    e->tag = tag;
    e->caller = caller;
    g_wlog_n++;
    g_issued++;
    pthread_mutex_unlock(&g_wp_lock);
}

uint64_t kot_write_log_total(void)
{
    pthread_mutex_lock(&g_wp_lock);
    uint64_t n = g_wlog_n;
    pthread_mutex_unlock(&g_wp_lock);
    return n;
}

void kot_write_log_dump(FILE *out)
{
    if (out == NULL)
        out = stderr;
    pthread_mutex_lock(&g_wp_lock);
    uint64_t n = g_wlog_n, first = n > KOT_WLOG ? n - KOT_WLOG : 0;
    fprintf(out, "# kot write log: %llu writes issued, showing %llu (seq space bar offset value width rc tag caller)\n",
            (unsigned long long)n, (unsigned long long)(n - first));
    for (uint64_t s = first; s < n; s++) {
        const struct kot_wlog *e = &g_wlog[s % KOT_WLOG];
        Dl_info di;
        const char *who = "?";
        if (e->caller != NULL && dladdr(e->caller, &di) && di.dli_sname)
            who = di.dli_sname;
        fprintf(out, "%llu %s %u 0x%06x 0x%08x %u %d %s %s\n", (unsigned long long)e->seq,
                e->space == KOT_WP_CFG ? "cfg" : "mmio", e->bar, e->offset, e->value, e->width, e->rc,
                e->tag ? e->tag : "?", who);
    }
    pthread_mutex_unlock(&g_wp_lock);
}

void kot_wp_stats(struct kot_wp_stats *out)
{
    pthread_mutex_lock(&g_wp_lock);
    kot_wp_load_allowlist_locked();
    out->mmio_denied = g_mmio_denied;
    out->cfg_denied = g_cfg_denied;
    out->logged = g_logged;
    out->issued = g_issued;
    out->allowlist_entries = g_allow_n;
    pthread_mutex_unlock(&g_wp_lock);
}

void kot_wp_reset_for_tests(void)
{
    pthread_mutex_lock(&g_wp_lock);
    g_allow_loaded = 0;
    g_allow_n = 0;
    g_mmio_denied = g_cfg_denied = g_logged = g_issued = 0;
    g_wlog_n = 0;
    memset(g_seen, 0, sizeof(g_seen));
    pthread_mutex_unlock(&g_wp_lock);
}
