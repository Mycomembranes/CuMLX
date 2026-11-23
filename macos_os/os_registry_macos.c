/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/os_registry_macos.c -- macOS replacement for
 * kernel-open/nvidia/os-registry.c (OGK 570.190).
 *
 * Registry values come from an in-memory table filled from the environment
 * variable KOT_RM_REGISTRY="Key=Value;Key=Value" (no module parameters, no
 * files). Values that parse completely as an unsigned integer (strtoul base 0)
 * are dwords, everything else is a string. A key that is not in the table is
 * "not found".
 *
 * os_registry_init() does not touch the device. It loads the table and hands
 * each entry to RM with rm_write_registry_dword/_string (what the Linux file
 * does with its NVreg_* parameters), so the RM-side readers in
 * arch/nvalloc/unix/src/registry.c answer from the same table.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvtypes.h"
#include "nvstatus.h"
#include "os-interface.h"
#include "nv.h"

#include "macos_os_priv.h"

#define KOT_REG_MAX 128
#define KOT_REG_KEY 64
#define KOT_REG_VAL 256

typedef struct {
    char key[KOT_REG_KEY];
    char str[KOT_REG_VAL];
    NvU32 dword;
    int is_dword;
} kot_reg_entry_t;

static pthread_mutex_t g_reg_lock = PTHREAD_MUTEX_INITIALIZER;
static kot_reg_entry_t g_reg[KOT_REG_MAX];
static int g_reg_count;

void kot_registry_clear(void)
{
    pthread_mutex_lock(&g_reg_lock);
    g_reg_count = 0;
    memset(g_reg, 0, sizeof(g_reg));
    pthread_mutex_unlock(&g_reg_lock);
}

static void kot_reg_put(const char *k, size_t klen, const char *v, size_t vlen)
{
    if (klen == 0 || klen >= KOT_REG_KEY || vlen >= KOT_REG_VAL)
        return;
    kot_reg_entry_t *e = NULL;
    for (int i = 0; i < g_reg_count; i++) {
        if (strlen(g_reg[i].key) == klen && memcmp(g_reg[i].key, k, klen) == 0) {
            e = &g_reg[i];
            break;
        }
    }
    if (e == NULL) {
        if (g_reg_count >= KOT_REG_MAX)
            return;
        e = &g_reg[g_reg_count++];
    }
    memset(e, 0, sizeof(*e));
    memcpy(e->key, k, klen);
    memcpy(e->str, v, vlen);
    char *end = NULL;
    unsigned long n = strtoul(e->str, &end, 0);
    e->is_dword = (vlen > 0 && end != NULL && *end == '\0');
    e->dword = (NvU32)n;
}

int kot_registry_load_env(void)
{
    const char *env = getenv("KOT_RM_REGISTRY");
    pthread_mutex_lock(&g_reg_lock);
    g_reg_count = 0;
    memset(g_reg, 0, sizeof(g_reg));
    for (const char *p = env; p && *p;) {
        const char *end = strchr(p, ';');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        const char *eq = memchr(p, '=', len);
        if (eq != NULL)
            kot_reg_put(p, (size_t)(eq - p), eq + 1, len - (size_t)(eq - p) - 1);
        p = end ? end + 1 : p + len;
    }
    int n = g_reg_count;
    pthread_mutex_unlock(&g_reg_lock);
    return n;
}

int kot_registry_read_dword(const char *key, uint32_t *value)
{
    int rc = -1;
    pthread_mutex_lock(&g_reg_lock);
    for (int i = 0; i < g_reg_count; i++) {
        if (g_reg[i].is_dword && strcmp(g_reg[i].key, key) == 0) {
            *value = g_reg[i].dword;
            rc = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_reg_lock);
    return rc;
}

int kot_registry_read_string(const char *key, char *buf, size_t buf_len)
{
    int rc = -1;
    pthread_mutex_lock(&g_reg_lock);
    for (int i = 0; i < g_reg_count; i++) {
        if (strcmp(g_reg[i].key, key) == 0) {
            if (buf && buf_len)
                snprintf(buf, buf_len, "%s", g_reg[i].str);
            rc = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_reg_lock);
    return rc;
}

NV_STATUS NV_API_CALL os_registry_init(void)
{
    kot_registry_load_env();
    kot_reg_entry_t copy[KOT_REG_MAX];
    pthread_mutex_lock(&g_reg_lock);
    int n = g_reg_count;
    memcpy(copy, g_reg, sizeof(copy));
    pthread_mutex_unlock(&g_reg_lock);
    for (int i = 0; i < n; i++) {
        if (copy[i].is_dword)
            rm_write_registry_dword(NULL, NULL, copy[i].key, copy[i].dword);
        else
            rm_write_registry_string(NULL, NULL, copy[i].key, copy[i].str,
                                     (NvU32)strlen(copy[i].str) + 1);
    }
    return NV_OK;
}
