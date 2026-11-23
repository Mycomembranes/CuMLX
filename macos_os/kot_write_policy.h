/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_write_policy.h -- local write policy at the two device
 * write choke points (shims/shim_io.c for MMIO, kot_cfg_write in kot_device.c
 * for PCI config, which os_pci_macos.c and shims/shim_pci.c both use), plus an
 * in-memory ring log of every write actually issued. Plain C.
 *
 * Compiled-in table, first match wins, three verdicts:
 *   DENY       never issued; the caller sees an error (RM config writes:
 *              NV_ERR_NOT_SUPPORTED; Linux pci_write_config_*: -EPERM; void MMIO
 *              writers: no write, one stderr line per address); counted.
 *   ALLOW_LOG  issued and logged; one stderr line per distinct address.
 *   ALLOW      issued (and, like every issued write, in the ring log).
 *
 * PCI config (decided on the aligned dword that would be issued, so a byte or
 * word RMW that rewrites a protected register is denied too):
 *   DENY 0x04 command, 0x10..0x27 BARs (per dword), 0x30 expansion ROM, the PM
 *   capability's PMCSR, MSI control, MSI-X control, PCIe Device Control and
 *   Link Control (capability offsets from the walk in kot_device.c; if the walk
 *   failed every config write is denied), and every extended-config write
 *   (offset >= 0x100: AER, Resizable BAR, ...).
 * MMIO (BAR0 offsets):
 *   DENY 0x000200 NV_PMC_ENABLE, 0x000600 NV_PMC_DEVICE_ENABLE(0),
 *        0x1103c0 NV_PGSP_FALCON_ENGINE, 0x8403c0 NV_PSEC_FALCON_ENGINE,
 *        0x8f03c0 FSP falcon ENGINE (falcon base 0x8f0000 + 0x3c0),
 *        0x088000..0x088fff the BAR0 mirror of PCI config space.
 *   ALLOW_LOG the rest of the GSP (0x110000..0x111fff), SEC2 (0x840000..0x841fff)
 *        and FSP (0x8f0000..0x8f3fff) falcon windows, and the WPR2 bounds
 *        (0x1fa824, 0x1fa828).
 * Allowlist: KOT_WRITE_ALLOWLIST=<file>, one entry per line, '#' comments:
 *   <bar>:<offset>   e.g. "0:0x200" lifts a DENY for that BAR dword
 *   cfg:<offset>     e.g. "cfg:0x04" lifts the DENY of the protected config
 *                    register that starts at that offset (BAR dwords: 0x10,
 *                    0x14, ... 0x24; extended config: the exact dword)
 * The allowlist never lifts the deny-all that follows a failed capability walk
 * or kot_fatal(). Every write is denied once kot_fatal() has been entered.
 */
#ifndef KOT_WRITE_POLICY_H
#define KOT_WRITE_POLICY_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum kot_wp_verdict { KOT_WP_ALLOW = 0, KOT_WP_DENY = 1, KOT_WP_ALLOW_LOG = 2 };
enum kot_wp_space { KOT_WP_MMIO = 0, KOT_WP_CFG = 1 };

/* Decide on one MMIO dword write (bar = KPI bar index, offset = BAR offset). */
enum kot_wp_verdict kot_wp_check_mmio(uint32_t bar, uint32_t offset, const char **rule);
/* Decide on one config dword write at aligned_offset. */
enum kot_wp_verdict kot_wp_check_cfg(uint32_t aligned_offset, const char **rule);

/* Record a denied write: counter + one stderr line per (space, bar, offset). */
void kot_wp_note_denied(enum kot_wp_space space, uint32_t bar, uint32_t offset, uint32_t value,
                        const char *rule, const char *tag);
/* Record an ALLOW_LOG write (one stderr line per address). */
void kot_wp_note_logged(enum kot_wp_space space, uint32_t bar, uint32_t offset, uint32_t value,
                        const char *rule, const char *tag);

/* Ring log (4096 entries) of every write actually issued. caller = return
 * address of the accessor's caller (resolved with dladdr by the dump). */
void kot_write_log_record(enum kot_wp_space space, uint32_t bar, uint32_t offset, uint32_t value,
                          uint32_t width, const char *tag, const void *caller, int rc);
void kot_write_log_dump(FILE *out);
uint64_t kot_write_log_total(void);   /* writes recorded since start (not capped) */

struct kot_wp_stats {
    uint64_t mmio_denied, cfg_denied, logged, issued;
    uint64_t allowlist_entries;
};
void kot_wp_stats(struct kot_wp_stats *out);

/* Re-read KOT_WRITE_ALLOWLIST and clear counters/once-flags (host tests). */
void kot_wp_reset_for_tests(void);

#ifdef __cplusplus
}
#endif
#endif /* KOT_WRITE_POLICY_H */
