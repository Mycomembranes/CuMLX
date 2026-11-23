/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/macos_os_priv.h -- private header of the macOS userspace
 * replacement for the OGK Linux OS layer (kernel-open/nvidia/os-interface.c,
 * os-pci.c, os-mlock.c, os-registry.c, the RM-facing part of nv.c, and
 * nv-nano-timer.c).
 *
 * Built only by the opt-in closure build (build.sh KOT_RM_CLOSURE=1, OGK 570.*)
 * and the host tests, with the same flags as the RM objects plus
 * -DKOT_SAFE_SHIMS -DSHIM_MUTATION_PREGATE, so every definition is checked
 * against the real 570.190 prototypes in os-interface.h / nv.h. The closure link
 * uses the KOT_SAFE_SHIMS builds of shims/shim_{io,pci,dma,firmware}.c, never
 * the default-build objects.
 *
 * What is enforced (and where):
 *   - Identity (kot_device.h): no config or MMIO access until
 *     KOT_REGISTRY_ENTRY_ID equals the registry entry id the KPI reports for the
 *     one NVIDIA device it resolves; os_pci_init_handle returns NULL otherwise.
 *   - PCI config -> kot_cfg_read / kot_cfg_write (kot_device.c), the single
 *     config choke point for os_pci_macos.c and shims/shim_pci.c: write policy
 *     (kot_write_policy.h), shim_mutation_admit (SHIM_MUTATION_PREGATE is
 *     defined for this build, so it really asks cudaDriverKitKpiMutationAdmit),
 *     dword read-modify-write for byte/word writes with no write-back after a
 *     failed read, ring log of every issued write.
 *   - BAR MMIO -> registry tokens (kot_tokens.h): ioremap / pci_iomap /
 *     os_map_kernel_space return a non-canonical token for a range inside a BAR
 *     of the accepted device (NULL otherwise); shim_read* and shim_write*
 *     validate, bounds-check and decode it, apply the write policy and
 *     shim_mutation_admit, and call cudaDriverKitKpiReadReg32 / WriteReg32. RM's
 *     NV_PRIV_REG_RD/WR macros are routed there by override/nv-priv.h. A raw
 *     dereference of a token faults (trap runtime: KOT_TOKEN_DEREFERENCED).
 *   - DMA -> shim_dma.c (dma_alloc_coherent / dma_free_coherent / map / unmap /
 *     sync); the device IOVA is the "physical" address RM sees. Frees are
 *     quarantined unless kot_device_quiesced() (kot_lifecycle.h).
 *   - Firmware -> shim_firmware.c: absolute NVIDIA_KO_FIRMWARE_ROOT, pinned
 *     files and SHA256 only.
 *   - Fatal paths -> kot_fatal (kot_lifecycle.h): parks, never exits, while DMA
 *     or a device handle may be live.
 *
 * Interrupts (polling model; there is no MSI/INTx delivery on this path): a
 * harness records the ISR and bottom half with kot_irq_register() (normally thin
 * wrappers around rm_isr() and rm_isr_bh()). kot_poll_interrupts() runs the ISR
 * on the calling thread with os_is_isr() NV_TRUE, EXCLUSIVE against every thread
 * holding an os_acquire_spinlock() lock (the model of spin_lock_irqsave: a
 * spinlock holder has "interrupts off"; see the gate in os_interface_macos.c),
 * then the bottom half outside the gate if the ISR asked for it. It gives up
 * (returns -3) if the gate cannot be entered within 10 ms. The optional poll
 * thread (kot_irq_poll_thread_start) calls it at a fixed period; it is NOT
 * started by default. At the first accepted device access the layer reads
 * (never writes) PCI command INTx-disable and the MSI / MSI-X enable bits and
 * reports them (kot_device_irq_config).
 *
 * Timers (kot_timer_macos.c): one timer thread; expiries are delivered on the
 * single work-queue worker (rm_run_nano_timer_callback, rm_run_rc_callback).
 *
 * Refused functions print one stderr note per function and return
 * NV_ERR_NOT_SUPPORTED (or the type's "unavailable" value); they never call
 * the KPI. See shims/macos_os/COVERAGE.md.
 */
#ifndef KOT_MACOS_OS_PRIV_H
#define KOT_MACOS_OS_PRIV_H

#include <stdint.h>
#include <stddef.h>

/* ---- refusal helper ---------------------------------------------------- */
void kot_os_note_refused(const char *fn, int *once);
#define KOT_REFUSED()                                    \
    do {                                                 \
        static int kot_once__;                           \
        kot_os_note_refused(__func__, &kot_once__);      \
    } while (0)

/* ---- page geometry RM sees (matches the shim linux/mm.h PAGE_SHIFT) ----- */
#define KOT_OS_PAGE_SHIFT 12u
#define KOT_OS_PAGE_SIZE  (1u << KOT_OS_PAGE_SHIFT)

/* ---- ISR context (os_is_isr) ------------------------------------------- */
int kot_os_in_isr(void);
void kot_os_set_in_isr(int on);

/* ---- interrupts (no MSI): record + poll (model above) ------------------ */
typedef int  (*kot_isr_fn)(void *ctx, int *need_bottom_half);
typedef void (*kot_bh_fn)(void *ctx);
/* Record the handler pair; replaces any previous one. Returns 0. */
int kot_irq_register(kot_isr_fn isr, kot_bh_fn bh, void *ctx);
void kot_irq_unregister(void);
/* Run the recorded ISR once (and the BH if requested). Returns 1 if the ISR
 * reported the interrupt as handled, 0 if not, -1 if nothing is registered,
 * -2 after kot_fatal, -3 if the spinlock gate could not be entered in 10 ms. */
int kot_poll_interrupts(void);
/* Periodic poll thread calling kot_poll_interrupts(); not started by default. */
int kot_irq_poll_thread_start(uint64_t period_us);
void kot_irq_poll_thread_stop(void);
uint64_t kot_irq_poll_cycles(void);
/* The gate itself (os_interface_macos.c): enter exclusively (1) or time out (0). */
int kot_irq_gate_enter_isr(uint64_t timeout_ns);
void kot_irq_gate_exit_isr(void);

/* ---- PCI: the single KPI device ---------------------------------------- */
/* Location answered by os_pci_init_handle(); default 0:00:00.0. */
void kot_pci_set_gpu_location(uint32_t domain, uint8_t bus, uint8_t slot, uint8_t function);

/* ---- registry (in-memory table, KOT_RM_REGISTRY="Key=Value;Key=Value") --- */
/* Return 0 and the value if the key is present, -1 ("not found") otherwise. */
int kot_registry_read_dword(const char *key, uint32_t *value);
int kot_registry_read_string(const char *key, char *buf, size_t buf_len);
/* Re-read KOT_RM_REGISTRY (tests); os_registry_init() calls it once. */
int kot_registry_load_env(void);
void kot_registry_clear(void);

/* ---- work queue (single FIFO worker thread) ---------------------------- */
/* The worker calls rm_execute_work_item(NULL, data) for every RM item and
 * fn(arg) for every kot_work_post() item, in FIFO order. */
uint64_t kot_work_executed_count(void);
uint64_t kot_work_refused_count(void);
int kot_work_post(void (*fn)(void *), void *arg);   /* 0, or -1 if refused */
int kot_work_on_worker(void);                        /* 1 on the worker thread */
/* Teardown: refuse new items, drain the queue, end the worker. Returns the
 * number of items refused so far, or -1 if called on the worker. */
int kot_work_queue_stop(void);

/* ---- timers (kot_timer_macos.c) ---------------------------------------- */
uint64_t kot_timer_fired_count(void);
void kot_timer_shutdown(void);   /* cancel every timer, end the timer thread */

/* ---- nv_alloc_pages registry (nv_macos.c) ------------------------------ */
uint64_t kot_alloc_bad_free_count(void);

/* ---- shim entry points reused from shims/shim_io.c (KOT_SAFE_SHIMS) ----
 * (shim_dma.c's dma_* come from the shim <linux/dma-mapping.h> in nv_macos.c) */
void *ioremap(uint64_t offset, unsigned long size);
void iounmap(volatile void *addr);

#endif /* KOT_MACOS_OS_PRIV_H */
