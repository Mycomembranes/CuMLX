/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_device.h -- the one device this layer may touch: identity
 * gate, BAR geometry, and the single PCI config-space choke point.
 *
 * Plain C (no RM headers). Used by shims/shim_io.c, shims/shim_pci.c (both
 * built with KOT_SAFE_SHIMS) and shims/macos_os/os_pci_macos.c.
 *
 * ---- identity gate ----------------------------------------------------------
 * No device access (config, MMIO) happens until kot_device_ready() has accepted
 * the device, once per process:
 *   1. KOT_REGISTRY_ENTRY_ID must name the card: exactly 16 hex digits, not 0.
 *   2. cudaDriverKitKpiDeviceCount(0) must report exactly one NVIDIA device
 *      (the KPI calls carry no device handle, so with two cards the MMIO target
 *      of the bridge could differ from the one resolved here).
 *   3. cudaDriverKitKpiResolveDevice(NVIDIA_DRIVERKIT, -1, 0) must succeed with
 *      vendor 0x10de and registry_entry_id equal to KOT_REGISTRY_ENTRY_ID.
 * A refusal is sticky: every later access is refused too (reads all-ones,
 * writes dropped, config helpers return KOT_DEV_REFUSED). Acceptance marks the
 * device as resolved for kot_fatal (kot_lifecycle_note_device_resolved).
 *
 * ---- BARs RM sees -----------------------------------------------------------
 * The KPI gives no CPU physical address, so the layer hands RM synthetic,
 * distinct "physical" bases that are never dereferenced (os_map_kernel_space /
 * ioremap turn them into tokens): slot 0 = PCI BAR0 (registers, KPI bar 0,
 * size bar0_length), slot 1 = PCI BAR1 (framebuffer, KPI bar 1, size
 * bar1_length). No other BAR is mappable (the KPI reports no size for it).
 */
#ifndef KOT_DEVICE_H
#define KOT_DEVICE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KOT_DEV_OK            0
#define KOT_DEV_REFUSED      (-1)   /* identity gate refused / not accepted */
#define KOT_DEV_IO_ERROR     (-2)   /* the KPI call failed */
#define KOT_DEV_DENIED       (-3)   /* the write policy denied the write */
#define KOT_DEV_NOT_ADMITTED (-4)   /* shim_mutation_admit refused */
#define KOT_DEV_BAD_ARG      (-5)   /* offset/width out of range or misaligned */

#define KOT_DEV_NUM_BARS 2u
#define KOT_DEV_BAR0_PHYS 0x0000004000000000ull   /* synthetic, never dereferenced */
#define KOT_DEV_BAR1_PHYS 0x0000008000000000ull

/* 1 if the device is accepted (runs the identity gate on first call). */
int kot_device_ready(void);
/* Why the gate refused ("" if accepted or not yet run). */
const char *kot_device_refusal(void);
/* Accepted identity (0 if not accepted). */
uint64_t kot_device_registry_entry_id(void);
uint16_t kot_device_pci_vendor(void);
uint16_t kot_device_pci_device(void);

/* BAR slot geometry; 0 and the values, or -1 (no such BAR / not accepted). */
int kot_device_bar(unsigned slot, uint64_t *phys_base, uint64_t *size);
/* The KPI bar index and BAR offset of [phys, phys+len); 0 or -1 if the range is
 * not entirely inside one BAR of the accepted device. */
int kot_device_bar_for_phys(uint64_t phys, uint64_t len, unsigned *slot, uint64_t *offset);

/* PCI config space (4 KiB). The ONLY config path of the layer. */
int kot_cfg_read32(uint32_t aligned_offset, uint32_t *out);   /* *out all-ones on failure */
int kot_cfg_read(uint32_t offset, uint32_t width, uint32_t *out);
/* Byte/word writes are an aligned dword read-modify-write; the policy decides
 * on the dword that would be issued; nothing is written after a failed read. */
int kot_cfg_write(uint32_t offset, uint32_t width, uint32_t value, const char *tag);

/* Capability offsets found by walking the list through kot_cfg_read32 (0 if
 * absent). ok == 0: the walk failed (no list, a loop, a read error, or no PCIe
 * capability), and the policy then denies every config write. */
struct kot_pci_caps {
    int ok;
    uint32_t pm, msi, msix, pcie;
    char why[96];
};
const struct kot_pci_caps *kot_device_caps(void);

/* Read-only interrupt configuration, read once at the first accepted access:
 * PCI command INTx-disable (bit 10), MSI enable (MSI control bit 0), MSI-X
 * enable (control bit 15). -1 = unknown. Printed to stderr once. */
struct kot_irq_config {
    int read;          /* 1 once read */
    int intx_disabled, msi_enabled, msix_enabled;
};
const struct kot_irq_config *kot_device_irq_config(void);

/* Host tests only: forget the gate verdict and the capability walk. */
void kot_device_reset_for_tests(void);

#ifdef __cplusplus
}
#endif
#endif /* KOT_DEVICE_H */
