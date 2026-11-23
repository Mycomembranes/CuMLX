/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_tokens.h -- BAR mapping tokens (shims/shim_io.c, built with
 * KOT_SAFE_SHIMS). Plain C.
 *
 * ioremap(), pci_iomap() and os_map_kernel_space() return
 *     KOT_TOKEN_BASE | (index << 40) | offset_base
 * where index names an entry of a registry (BAR slot, BAR offset of the mapping
 * start, mapping size). KOT_TOKEN_BASE sets bit 52, so a token is a
 * non-canonical address: a raw CPU dereference faults (the trap-stub runtime
 * reports it as KOT_TOKEN_DEREFERENCED). RM adds register offsets to the token;
 * the low 40 bits are therefore the BAR offset of the access.
 *
 * Every access through shim_read* / shim_write* validates the token: the top
 * bits must be exactly KOT_TOKEN_BASE, the index must name a live entry (indices
 * are never reused, so an access through a token after iounmap() /
 * os_unmap_kernel_space() can never alias a newer mapping or BAR0), and the
 * access [offset, offset+width) must lie inside the mapping and inside the BAR
 * size the KPI reported. A refused access reads all-ones, writes nothing, prints
 * one stderr line per kind of refusal and is counted. NULL is not BAR0.
 */
#ifndef KOT_TOKENS_H
#define KOT_TOKENS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KOT_TOKEN_BASE      (1ull << 52)
#define KOT_TOKEN_IDX_SHIFT 40
#define KOT_TOKEN_IDX_MAX   4095u
#define KOT_TOKEN_OFF_MASK  ((1ull << KOT_TOKEN_IDX_SHIFT) - 1)

/* Register a mapping of [bar_offset, bar_offset+size) of BAR slot (0/1). */
void *kot_token_map(unsigned slot, uint64_t bar_offset, uint64_t size);
/* Invalidate; 0, or -1 for an unknown / already-unmapped token. */
int kot_token_unmap(const volatile void *token);

struct kot_token_stats {
    uint64_t mapped, unmapped;
    uint64_t bad_token, stale_token, out_of_range, misaligned, device_refused, read_failed, unmap_unknown;
    uint64_t map_refused;
};
void kot_token_stats(struct kot_token_stats *out);
void kot_token_reset_for_tests(void);

#ifdef __cplusplus
}
#endif
#endif /* KOT_TOKENS_H */
