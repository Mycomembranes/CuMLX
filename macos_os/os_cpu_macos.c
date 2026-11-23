/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/os_cpu_macos.c -- userspace answers for the AArch64 system
 * registers RmInitCpuInfo() (platform/cpu.c) reads; see
 * override/kot_cpu_arm.h. MIDR reports implementer 0x61 (Apple) with part 0,
 * which RM treats as an unknown ARMv8 CPU (it prints a notice and continues).
 * CCSIDR is synthesised in the architectural encoding from sysctl
 * hw.cachelinesize / hw.l1dcachesize / hw.l2cachesize (8-way assumed), for
 * the level last selected with CSSELR.
 */
#include <stdint.h>
#include <sys/sysctl.h>

static uint32_t g_csselr;

static uint64_t kot_sysctl_u64(const char *name, uint64_t dflt)
{
    uint64_t v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, NULL, 0) != 0 || v == 0)
        return dflt;
    return v;
}

uint32_t kot_cpu_read_midr(void) { return 0x61u << 24; }

void kot_cpu_write_csselr(uint32_t val) { g_csselr = val; }

uint32_t kot_cpu_read_ccsidr(void)
{
    uint64_t line = kot_sysctl_u64("hw.cachelinesize", 64);
    uint64_t size = (g_csselr == 2) ? kot_sysctl_u64("hw.l2cachesize", 4u << 20)
                                    : kot_sysctl_u64("hw.l1dcachesize", 64u << 10);
    uint32_t assoc = 8;
    uint32_t log2line = 0;
    while ((1ull << log2line) < line)
        log2line++;
    uint64_t sets = size / (line * assoc);
    if (sets == 0)
        sets = 1;
    /* LineSize[2:0] = log2(line) - 4, Associativity[12:3] = assoc - 1, NumSets[27:13] = sets - 1 */
    return (uint32_t)(((sets - 1) & 0x7fff) << 13) | ((assoc - 1) << 3) | ((log2line - 4) & 7);
}
