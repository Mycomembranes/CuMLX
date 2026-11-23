/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/override/kot_cpu_arm.h -- force-included into
 * src/nvidia/src/kernel/platform/cpu.c only (closure build; see
 * tools/closure/closure_cflags.txt). RM's AArch64 RmInitCpuInfo() reads
 * MIDR_EL1 / CCSIDR_EL1 and writes CSSELR_EL1 with mrs/msr, which are EL1
 * system registers: in a macOS user process they raise SIGILL. These macros
 * route them to shims/macos_os/os_cpu_macos.c, which answers from sysctl.
 */
#include "cpu_arm_def.h"

#undef CP_READ_CCSIDR_REGISTER
#undef CP_READ_MIDR_REGISTER
#undef CP_WRITE_CSSELR_REGISTER

NvU32 kot_cpu_read_midr(void);
NvU32 kot_cpu_read_ccsidr(void);
void kot_cpu_write_csselr(NvU32 val);

#define CP_READ_CCSIDR_REGISTER()     kot_cpu_read_ccsidr()
#define CP_READ_MIDR_REGISTER()       kot_cpu_read_midr()
#define CP_WRITE_CSSELR_REGISTER(val) kot_cpu_write_csselr(val)
