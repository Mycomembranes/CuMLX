/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/override/nv-priv.h -- closure-build-only wrapper around the
 * OGK 570.190 arch/nvalloc/unix/include/nv-priv.h (KOT_RM_CLOSURE=1; this
 * directory is put first on the include path by tools/closure/closure_cflags.txt).
 *
 * The OGK header defines NV_PRIV_REG_RD/WR{08,16,32} as raw dereferences of a
 * BAR mapping. On this path the mapping returned by os_map_kernel_space() is a
 * shim_io.c token, not a CPU pointer, so every RM register access through these
 * macros (osDevReadReg* / osDevWriteReg* in os.c, which gpu_access.c's
 * GPU_REG_RD/WR paths use, and the direct PMC_BOOT_* reads in osinit.c /
 * osapi.c) is routed to shim_readb/w/l and shim_writeb/w/l, which decode the
 * token and call cudaDriverKitKpiReadReg32 / WriteReg32.
 */
#include_next <nv-priv.h>

#ifndef KOT_NV_PRIV_OVERRIDE_H
#define KOT_NV_PRIV_OVERRIDE_H

unsigned char  shim_readb(const volatile void *addr);
unsigned short shim_readw(const volatile void *addr);
unsigned int   shim_readl(const volatile void *addr);
void shim_writeb(unsigned char v, volatile void *addr);
void shim_writew(unsigned short v, volatile void *addr);
void shim_writel(unsigned int v, volatile void *addr);

#define KOT_REG_ADDR(b, o) ((volatile void *)((volatile char *)(b) + (o)))

#undef NV_PRIV_REG_WR08
#undef NV_PRIV_REG_WR16
#undef NV_PRIV_REG_WR32
#undef NV_PRIV_REG_RD08
#undef NV_PRIV_REG_RD16
#undef NV_PRIV_REG_RD32
#define NV_PRIV_REG_WR08(b,o,d) shim_writeb((unsigned char)(d), KOT_REG_ADDR(b, o))
#define NV_PRIV_REG_WR16(b,o,d) shim_writew((unsigned short)(d), KOT_REG_ADDR(b, o))
#define NV_PRIV_REG_WR32(b,o,d) shim_writel((unsigned int)(d), KOT_REG_ADDR(b, o))
#define NV_PRIV_REG_RD08(b,o)   ((NvU8)shim_readb(KOT_REG_ADDR(b, o)))
#define NV_PRIV_REG_RD16(b,o)   ((NvU16)shim_readw(KOT_REG_ADDR(b, o)))
#define NV_PRIV_REG_RD32(b,o)   ((NvU32)shim_readl(KOT_REG_ADDR(b, o)))

#endif /* KOT_NV_PRIV_OVERRIDE_H */
