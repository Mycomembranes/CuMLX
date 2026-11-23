/*
 * shims/shim_pci.c -- Linux PCI KPI for the nvidia-ko translator.
 *
 * KOT_SAFE_SHIMS variant (RM closure link, host tests): the struct pci_dev RM
 * probes is built from the identity-checked KPI device (kot_device.h): real
 * BAR0/BAR1 sizes from the KPI, distinct synthetic bases that ioremap() /
 * os_map_kernel_space() accept, no device at all when the identity gate
 * refuses. Config reads and writes go through kot_cfg_read / kot_cfg_write (the
 * single config choke point: write policy, dword read-modify-write for byte and
 * word writes, no write-back after a failed read, ring log). pci_iomap returns
 * a registry token (kot_tokens.h).
 * Without KOT_SAFE_SHIMS (default 570.190 / 595.71.05 build): the legacy code
 * below the #else, kept byte-identical; NOT fit to be put in front of a card.
 */
#if defined(KOT_SAFE_SHIMS)
#include <stdint.h>
#include <string.h>

#include <cuda_driverkit_kpi.h>
#include <linux/io.h>
#include <linux/pci.h>

#include "macos_os/kot_device.h"
#include "macos_os/kot_tokens.h"

static struct pci_dev g_nv_pci_dev;
static int g_nv_pci_ready;   /* 1 device present, -1 refused */

static void shim_pci_init_once(void)
{
    if (g_nv_pci_ready != 0)
        return;
    memset(&g_nv_pci_dev, 0, sizeof(g_nv_pci_dev));
    if (!kot_device_ready()) {
        g_nv_pci_ready = -1;
        return;
    }
    g_nv_pci_dev.vendor = kot_device_pci_vendor();
    g_nv_pci_dev.device = kot_device_pci_device();
    uint32_t dw;
    if (kot_cfg_read32(0x08, &dw) == KOT_DEV_OK) {
        g_nv_pci_dev.revision = (u8)dw;
        g_nv_pci_dev.class = dw >> 8;
    }
    if (kot_cfg_read32(0x2c, &dw) == KOT_DEV_OK) {
        g_nv_pci_dev.subsystem_vendor = (u16)dw;
        g_nv_pci_dev.subsystem_device = (u16)(dw >> 16);
    }
    for (unsigned s = 0; s < KOT_DEV_NUM_BARS; s++) {
        uint64_t base, size;
        if (kot_device_bar(s, &base, &size) != 0)
            continue;
        g_nv_pci_dev.resource[s].start = base;
        g_nv_pci_dev.resource[s].end = base + size - 1;
        g_nv_pci_dev.resource[s].flags = IORESOURCE_MEM;
    }
    g_nv_pci_ready = 1;
}

/* Host tests only: forget the probe-ready state (after kot_device_reset_for_tests). */
void shim_pci_reset_for_tests(void)
{
    g_nv_pci_ready = 0;
}

static int shim_pci_is_dev(const struct pci_dev *dev)
{
    shim_pci_init_once();
    return g_nv_pci_ready == 1 && (dev == 0 || dev == &g_nv_pci_dev);
}

int pci_enable_device(struct pci_dev *dev)
{
    return shim_pci_is_dev(dev) ? 0 : -19 /* -ENODEV */;
}

/* No config writes here: bus mastering / decode are left as the KPI owner set
 * them (the PCI command register is on the write-policy deny list). */
void pci_disable_device(struct pci_dev *dev) { (void)dev; }
void pci_set_master(struct pci_dev *dev) { (void)dev; }
int pci_request_regions(struct pci_dev *dev, const char *res_name) { (void)res_name; return shim_pci_is_dev(dev) ? 0 : -19; }
void pci_release_regions(struct pci_dev *dev) { (void)dev; }

static int shim_pci_rc(int rc)
{
    switch (rc) {
    case KOT_DEV_OK:      return 0;
    case KOT_DEV_DENIED:  return -1;    /* -EPERM */
    case KOT_DEV_BAD_ARG: return -22;   /* -EINVAL */
    case KOT_DEV_REFUSED: return -19;   /* -ENODEV */
    case KOT_DEV_NOT_ADMITTED: return -2000;   /* SHIM_MUTATION_DENIED */
    default:              return -5;    /* -EIO */
    }
}

static int shim_pci_read(const struct pci_dev *dev, int where, uint32_t width, uint32_t *out)
{
    if (!shim_pci_is_dev(dev) || where < 0) {
        *out = width == 4 ? 0xffffffffu : (1u << (width * 8)) - 1u;
        return -19;
    }
    return shim_pci_rc(kot_cfg_read((uint32_t)where, width, out));
}

int pci_read_config_byte(const struct pci_dev *dev, int where, u8 *val)
{
    uint32_t tmp;
    int rc = shim_pci_read(dev, where, 1, &tmp);
    if (val != 0)
        *val = (u8)tmp;
    return rc;
}

int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val)
{
    uint32_t tmp;
    int rc = shim_pci_read(dev, where, 2, &tmp);
    if (val != 0)
        *val = (u16)tmp;
    return rc;
}

int pci_read_config_dword(const struct pci_dev *dev, int where, u32 *val)
{
    uint32_t tmp;
    int rc = shim_pci_read(dev, where, 4, &tmp);
    if (val != 0)
        *val = (u32)tmp;
    return rc;
}

static int shim_pci_write(const struct pci_dev *dev, int where, uint32_t width, uint32_t val, const char *tag)
{
    if (!shim_pci_is_dev(dev) || where < 0)
        return -19;
    return shim_pci_rc(kot_cfg_write((uint32_t)where, width, val, tag));
}

int pci_write_config_byte(const struct pci_dev *dev, int where, u8 val)
{
    return shim_pci_write(dev, where, 1, val, "pci_write_config_byte");
}

int pci_write_config_word(const struct pci_dev *dev, int where, u16 val)
{
    return shim_pci_write(dev, where, 2, val, "pci_write_config_word");
}

int pci_write_config_dword(const struct pci_dev *dev, int where, u32 val)
{
    return shim_pci_write(dev, where, 4, val, "pci_write_config_dword");
}

void __iomem *pci_iomap(struct pci_dev *dev, int bar, unsigned long max_len)
{
    uint64_t base, size;
    if (!shim_pci_is_dev(dev) || bar < 0 || kot_device_bar((unsigned)bar, &base, &size) != 0)
        return 0;
    if (max_len != 0 && max_len < size)
        size = max_len;
    return (void __iomem *)kot_token_map((unsigned)bar, 0, size);
}

void pci_iounmap(struct pci_dev *dev, void __iomem *addr)
{
    (void)dev;
    iounmap(addr);
}

int pci_register_driver(struct pci_driver *drv)
{
    shim_pci_init_once();
    if (g_nv_pci_ready != 1)
        return -19;   /* identity gate refused: no device to probe */
    if (drv != 0 && drv->probe != 0) {
        return drv->probe(&g_nv_pci_dev, drv->id_table);
    }
    return 0;
}

void pci_unregister_driver(struct pci_driver *drv) { (void)drv; }
void *pci_get_drvdata(const struct pci_dev *dev) { return dev ? dev->driver_data : 0; }
void pci_set_drvdata(struct pci_dev *dev, void *data) { if (dev) dev->driver_data = data; }

#else /* !KOT_SAFE_SHIMS: legacy default-build variant, unchanged */
#include <stdint.h>
#include <string.h>

#include <cuda_driverkit_kpi.h>
#include <linux/pci.h>
#include <shim_mutation_admission.h>

static struct pci_dev g_nv_pci_dev;
static int g_nv_pci_ready;

static void shim_pci_init_once(void)
{
    if (g_nv_pci_ready) {
        return;
    }
    CudaDriverKitDeviceInfo info;
    memset(&info, 0, sizeof(info));
    int rc = cudaDriverKitKpiResolveDevice(CUDK_BACKEND_NVIDIA_DRIVERKIT, -1, 0, &info);
    if (rc != 0) {
        uint16_t vendor = 0x10de;
        uint16_t device = 0;
        uint32_t max_selector = 0;
        (void)cudaDriverKitKpiProbe(&vendor, &device, &max_selector);
        info.pci_vendor = vendor;
        info.pci_device = device;
        info.bar0_length = 0x01000000;
        info.bar1_length = 0x10000000;
    }
    memset(&g_nv_pci_dev, 0, sizeof(g_nv_pci_dev));
    g_nv_pci_dev.vendor = info.pci_vendor ? info.pci_vendor : 0x10de;
    g_nv_pci_dev.device = info.pci_device;
    g_nv_pci_dev.resource[0].start = 0;
    g_nv_pci_dev.resource[0].end = (info.bar0_length ? info.bar0_length : 0x01000000) - 1;
    g_nv_pci_dev.resource[1].start = 0;
    g_nv_pci_dev.resource[1].end = (info.bar1_length ? info.bar1_length : 0x10000000) - 1;
    g_nv_pci_ready = 1;
}

int pci_enable_device(struct pci_dev *dev)
{
    (void)dev;
    shim_pci_init_once();
    return 0;
}

void pci_disable_device(struct pci_dev *dev) { (void)dev; }
void pci_set_master(struct pci_dev *dev) { (void)dev; }
int pci_request_regions(struct pci_dev *dev, const char *res_name) { (void)dev; (void)res_name; return 0; }
void pci_release_regions(struct pci_dev *dev) { (void)dev; }

int pci_read_config_byte(const struct pci_dev *dev, int where, u8 *val)
{
    (void)dev;
    uint32_t tmp = 0;
    int rc = cudaDriverKitKpiPciReadConfig((uint32_t)where, 1, &tmp);
    if (rc == 0 && val != 0) {
        *val = (u8)tmp;
    }
    return rc;
}

int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val)
{
    (void)dev;
    uint32_t tmp = 0;
    int rc = cudaDriverKitKpiPciReadConfig((uint32_t)where, 2, &tmp);
    if (rc == 0 && val != 0) {
        *val = (u16)tmp;
    }
    return rc;
}

int pci_read_config_dword(const struct pci_dev *dev, int where, u32 *val)
{
    (void)dev;
    uint32_t tmp = 0;
    int rc = cudaDriverKitKpiPciReadConfig((uint32_t)where, 4, &tmp);
    if (rc == 0 && val != 0) {
        *val = (u32)tmp;
    }
    return rc;
}

int pci_write_config_byte(const struct pci_dev *dev, int where, u8 val)
{
    (void)dev;
    int admitted = shim_mutation_admit(CUDK_MUTATION_OP_PCI_WRITE_CONFIG);
    if (admitted != 0) {
        return admitted;
    }
    return cudaDriverKitKpiPciWriteConfig((uint32_t)where, 1, val);
}

int pci_write_config_word(const struct pci_dev *dev, int where, u16 val)
{
    (void)dev;
    int admitted = shim_mutation_admit(CUDK_MUTATION_OP_PCI_WRITE_CONFIG);
    if (admitted != 0) {
        return admitted;
    }
    return cudaDriverKitKpiPciWriteConfig((uint32_t)where, 2, val);
}

int pci_write_config_dword(const struct pci_dev *dev, int where, u32 val)
{
    (void)dev;
    int admitted = shim_mutation_admit(CUDK_MUTATION_OP_PCI_WRITE_CONFIG);
    if (admitted != 0) {
        return admitted;
    }
    return cudaDriverKitKpiPciWriteConfig((uint32_t)where, 4, val);
}

void __iomem *pci_iomap(struct pci_dev *dev, int bar, unsigned long max_len)
{
    (void)dev;
    (void)max_len;
    return (void __iomem *)(uintptr_t)((bar & 0xffff) + 1);
}

void pci_iounmap(struct pci_dev *dev, void __iomem *addr)
{
    (void)dev;
    (void)addr;
}

int pci_register_driver(struct pci_driver *drv)
{
    shim_pci_init_once();
    if (drv != 0 && drv->probe != 0) {
        return drv->probe(&g_nv_pci_dev, drv->id_table);
    }
    return 0;
}

void pci_unregister_driver(struct pci_driver *drv) { (void)drv; }
void *pci_get_drvdata(const struct pci_dev *dev) { return dev ? dev->driver_data : 0; }
void pci_set_drvdata(struct pci_dev *dev, void *data) { if (dev) dev->driver_data = data; }
#endif /* KOT_SAFE_SHIMS */
