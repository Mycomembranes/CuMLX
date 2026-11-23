/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_lifecycle.c -- see kot_lifecycle.h. Plain C.
 */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kot_lifecycle.h"

static volatile int g_quiesced;
static char g_quiesced_reason[256];
static volatile int g_device_resolved;
static volatile uint64_t g_dma_live, g_dma_quarantined;
static volatile int g_fatal_entered;
static volatile int g_fatal_printed;

static void kot_w(const char *s)
{
    if (s != NULL)
        (void)!write(2, s, strlen(s));
}

static void kot_wu(uint64_t v)
{
    char b[24];
    int i = (int)sizeof(b);
    b[--i] = 0;
    do {
        b[--i] = (char)('0' + (v % 10));
        v /= 10;
    } while (v != 0 && i > 0);
    kot_w(&b[i]);
}

void kot_device_quiesced_set(const char *reason)
{
    if (reason == NULL || reason[0] == 0) {
        kot_w("kot-lifecycle: kot_device_quiesced_set refused: a reason (what was proven) is required\n");
        return;
    }
    if (__atomic_load_n(&g_quiesced, __ATOMIC_ACQUIRE))
        return;
    snprintf(g_quiesced_reason, sizeof(g_quiesced_reason), "%s", reason);
    __atomic_store_n(&g_quiesced, 1, __ATOMIC_RELEASE);
    fprintf(stderr, "kot-lifecycle: device marked quiesced: %s\n", g_quiesced_reason);
}

int kot_device_quiesced(void) { return __atomic_load_n(&g_quiesced, __ATOMIC_ACQUIRE); }
const char *kot_device_quiesced_reason(void) { return kot_device_quiesced() ? g_quiesced_reason : ""; }

void kot_lifecycle_note_device_resolved(void) { __atomic_store_n(&g_device_resolved, 1, __ATOMIC_RELEASE); }
int kot_lifecycle_device_resolved(void) { return __atomic_load_n(&g_device_resolved, __ATOMIC_ACQUIRE); }

void kot_lifecycle_dma_counts(uint64_t live, uint64_t quarantined)
{
    __atomic_store_n(&g_dma_live, live, __ATOMIC_RELEASE);
    __atomic_store_n(&g_dma_quarantined, quarantined, __ATOMIC_RELEASE);
}
uint64_t kot_lifecycle_dma_live(void) { return __atomic_load_n(&g_dma_live, __ATOMIC_ACQUIRE); }
uint64_t kot_lifecycle_dma_quarantined(void) { return __atomic_load_n(&g_dma_quarantined, __ATOMIC_ACQUIRE); }

int kot_lifecycle_fatal_entered(void) { return __atomic_load_n(&g_fatal_entered, __ATOMIC_ACQUIRE); }

static int kot_fatal_aborts_env(void)
{
    const char *e = getenv("KOT_FATAL_ABORTS");
    return e != NULL && e[0] == '1' && e[1] == 0;
}

__attribute__((noreturn)) void kot_fatal(const char *what)
{
    __atomic_store_n(&g_fatal_entered, 1, __ATOMIC_RELEASE);
    uint64_t live = kot_lifecycle_dma_live(), quar = kot_lifecycle_dma_quarantined();
    int resolved = kot_lifecycle_device_resolved();
    int touched = live != 0 || quar != 0 || resolved;
    int aborts = kot_fatal_aborts_env();
    int first = __atomic_exchange_n(&g_fatal_printed, 1, __ATOMIC_ACQ_REL) == 0;
    if (first) {
        kot_w("kot-fatal: ");
        kot_w(what ? what : "(no reason)");
        kot_w(" [dma live ");
        kot_wu(live);
        kot_w(", quarantined ");
        kot_wu(quar);
        kot_w(", device handle resolved ");
        kot_w(resolved ? "yes" : "no");
        kot_w(", quiesced ");
        kot_w(kot_device_quiesced() ? "yes" : "no");
        kot_w("]\n");
    }
    if (aborts) {
        if (first)
            kot_w("kot-fatal: KOT_FATAL_ABORTS=1 (mock harness / host test): abort()\n");
        abort();
    }
    if (!touched) {
        if (first)
            kot_w("kot-fatal: nothing device-related was touched: abort()\n");
        abort();
    }
    if (first)
        kot_w("kot-fatal: device state may be owned by this process: PARKING this thread forever; "
              "DMA stays mapped. Recover the card (reset before release), then end the process.\n");
    for (;;)
        pause();
}

/* No-exit policy: a normal exit with DMA the card may target parks instead. */
static void kot_lifecycle_atexit(void)
{
    if (kot_device_quiesced())
        return;
    if (kot_lifecycle_dma_live() != 0 || kot_lifecycle_dma_quarantined() != 0)
        kot_fatal("process exit with live or quarantined DMA and the device not quiesced");
}

__attribute__((constructor)) static void kot_lifecycle_init(void)
{
    atexit(kot_lifecycle_atexit);
}
