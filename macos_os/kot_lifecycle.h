/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_lifecycle.h -- process lifecycle rules of the macOS RM
 * translator layer: DMA quarantine gate, no-exit policy, kot_fatal().
 *
 * Plain C (no RM headers): compiled with the RM flags in the closure build and
 * with plain flags in the host tests.
 *
 * Governing rule ("reset before release"): a process that may own device state
 * never unpins DMA memory the card may still target, and never exits while it
 * might. A failure parks the failing thread; it does not abort the process.
 *
 * ---- the "device quiesced" flag -----------------------------------------
 * Default: NOT quiesced. While it is clear, every DMA free (dma_free_coherent,
 * and nv_free_pages through it) moves the buffer to a quarantine list: it stays
 * allocated and mapped, the free returns success to RM, and
 * kot_quarantine_report() lists it. cudaDriverKitKpiDmaFree is called only
 * after the flag is set (kot_quarantine_release() frees the quarantine then).
 *
 * A caller may call kot_device_quiesced_set() ONLY after it has proven, for the
 * exact card this process drove (the registry entry id this layer verified),
 * one of:
 *   (a) the GSP falcon is held in reset AND WPR2 is down, both read back from
 *       the card after the last command this process issued
 *       (NV_PGSP_FALCON_ENGINE reset asserted; NV_PFB_PRI_MMU_WPR2_ADDR_HI == 0),
 *       so no engine on the card can still master DMA into host memory; or
 *   (b) a function-level reset of that card completed and GFW boot completion
 *       was confirmed afterwards (NV_THERM_I2CS_SCRATCH 0xAD00BC == 0xFF), so the
 *       card's DMA engines were reset after this process's last command; or
 *   (c) an operator attestation, recorded with the reason string, that the card
 *       was reset or powered off by other means.
 * The reason string is kept and printed in every report. Nothing in this tree
 * sets the flag except the host tests and the mock harness at its very end
 * (tools/rm_init_exercise.c, which never has a card behind it).
 *
 * ---- kot_fatal() ----------------------------------------------------------
 * Used by every trap stub (tools/closure/gen_trap_stubs.py), by os_bug_check and
 * by the fatal-signal handler. If any DMA allocation is live or quarantined, or
 * a device handle was ever resolved, it prints the reason once to stderr and
 * PARKS the calling thread forever (pause() loop): the process keeps its DMA
 * mappings and device handle until an operator recovers the card. Only when
 * nothing device-related was ever touched does it abort().
 * KOT_FATAL_ABORTS=1 (environment, read when kot_fatal runs) makes it abort()
 * in every case; for the mock harness and the host tests only, which must set
 * it explicitly.
 * After kot_fatal has been entered, the layer issues no further device writes
 * (kot_write_policy denies them), and the timer, interrupt-poll and work-queue
 * threads stop dispatching.
 *
 * Process exit: an atexit hook calls kot_fatal("process exit ...") when DMA is
 * live or quarantined and the quiesced flag is clear, so a normal return from
 * main parks instead of releasing the mappings.
 */
#ifndef KOT_LIFECYCLE_H
#define KOT_LIFECYCLE_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quiesced flag (see above). reason must be non-empty; the first call wins. */
void kot_device_quiesced_set(const char *reason);
int kot_device_quiesced(void);
const char *kot_device_quiesced_reason(void);

/* A device handle was resolved and accepted (identity verified). Sticky. */
void kot_lifecycle_note_device_resolved(void);
int kot_lifecycle_device_resolved(void);

/* DMA accounting, maintained by shim_dma.c: current live and quarantined
 * allocation counts. */
void kot_lifecycle_dma_counts(uint64_t live, uint64_t quarantined);
uint64_t kot_lifecycle_dma_live(void);
uint64_t kot_lifecycle_dma_quarantined(void);

/* 1 once kot_fatal() has been entered by any thread. */
int kot_lifecycle_fatal_entered(void);

/* Park or abort (see above). Async-signal tolerant: write(2), atomics, pause(). */
__attribute__((noreturn)) void kot_fatal(const char *what);

/* Implemented by shim_dma.c (KOT_SAFE_SHIMS): print every quarantined buffer
 * to out (NULL: stderr) and return the number listed. */
uint64_t kot_quarantine_report(FILE *out);
/* Free every quarantined buffer through cudaDriverKitKpiDmaFree. Refused
 * (returns -1, frees nothing) unless kot_device_quiesced(). Returns the number
 * freed. */
int64_t kot_quarantine_release(void);

#ifdef __cplusplus
}
#endif
#endif /* KOT_LIFECYCLE_H */
