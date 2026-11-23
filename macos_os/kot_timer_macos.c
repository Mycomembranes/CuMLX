/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/kot_timer_macos.c -- macOS replacement for
 * kernel-open/nvidia/nv-nano-timer.c (nv_create/start/cancel/destroy_nano_timer)
 * and the RC timer of kernel-open/nvidia/nv.c (nv_start_rc_timer /
 * nv_stop_rc_timer), so RM's osCreateNanoTimer path (os.c) and its 1 Hz
 * services (osRun1HzCallbacksNow via rm_run_rc_callback: GSP log polling,
 * watchdogs) actually run.
 *
 * Model: ONE timer thread keeps every armed timer and sleeps on the monotonic
 * clock until the earliest deadline. An expiry is not run on the timer thread:
 * it is posted to the single work-queue worker (kot_work_post), which calls
 *   nano timer: rm_run_nano_timer_callback(NULL, nv, pTmrEvent)
 *               (RM then queues its own tmrEventServiceTimer work item);
 *   RC timer:   rm_run_rc_callback(NULL, nv), re-armed 1 s later while it
 *               returns NV_OK and nv->rc_timer_enabled is set (as nv.c does).
 * So every RM callback runs on the same thread as RM's work items. The stack
 * argument is NULL: on arm64 NV_ENTER_RM_RUNTIME does not use it (osapi.h).
 *
 * Cancel semantics (hrtimer_cancel / del_timer_sync): a cancelled timer's
 * pending expiry is dropped (generation check); if its callback is running on
 * the worker, a caller on another thread waits for it to finish; a caller on
 * the worker itself does not wait. Destroyed timers are kept (never freed)
 * because a posted expiry may still refer to them; RM creates only a handful.
 * Nothing is dispatched after kot_fatal(). kot_timer_shutdown() cancels every
 * timer and ends the timer thread (teardown).
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nvtypes.h"
#include "nvstatus.h"
#include "os-interface.h"
#include "nv.h"

#include "macos_os_priv.h"
#include "kot_lifecycle.h"

#define KOT_TIMER_MAGIC 0x6b6f746du   /* 'kotm' */
#define KOT_RC_PERIOD_NS 1000000000ull

enum { KOT_TIMER_NANO = 1, KOT_TIMER_RC = 2 };

struct nv_nano_timer {
    NvU32 magic;
    int kind;
    nv_state_t *nv;
    void *pTmrEvent;
    uint64_t deadline;     /* monotonic ns */
    int armed;
    int destroyed;
    uint64_t gen;          /* bumped by start/cancel: a posted expiry carries it */
    int running;           /* its callback is executing on the worker */
    uint64_t fired;
    struct nv_nano_timer *next;
};

struct kot_fire {
    struct nv_nano_timer *t;
    uint64_t gen;
};

static pthread_mutex_t g_tm_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tm_cv = PTHREAD_COND_INITIALIZER;     /* timer list changed */
static pthread_cond_t g_tm_done = PTHREAD_COND_INITIALIZER;   /* a callback finished */
static struct nv_nano_timer *g_timers;
static int g_tm_started, g_tm_exit;
static pthread_t g_tm_thread;
static uint64_t g_tm_fired, g_tm_dropped;

static uint64_t kot_now_ns(void) { return (uint64_t)clock_gettime_nsec_np(CLOCK_MONOTONIC); }

static void kot_timer_fire(void *arg);

static void *kot_timer_main(void *arg)
{
    (void)arg;
    pthread_mutex_lock(&g_tm_lock);
    while (!g_tm_exit) {
        uint64_t now = kot_now_ns(), next = UINT64_MAX;
        for (struct nv_nano_timer *t = g_timers; t != NULL; t = t->next) {
            if (!t->armed)
                continue;
            if (t->deadline <= now) {
                t->armed = 0;
                struct kot_fire *f = malloc(sizeof(*f));
                if (f != NULL && !kot_lifecycle_fatal_entered()) {
                    f->t = t;
                    f->gen = t->gen;
                    if (kot_work_post(kot_timer_fire, f) != 0) {
                        free(f);
                        g_tm_dropped++;
                    }
                } else {
                    free(f);
                    g_tm_dropped++;
                }
            } else if (t->deadline < next) {
                next = t->deadline;
            }
        }
        if (next == UINT64_MAX) {
            pthread_cond_wait(&g_tm_cv, &g_tm_lock);
        } else {
            uint64_t d = next - now;
            struct timespec rel = { (time_t)(d / 1000000000ull), (long)(d % 1000000000ull) };
            pthread_cond_timedwait_relative_np(&g_tm_cv, &g_tm_lock, &rel);
        }
    }
    pthread_mutex_unlock(&g_tm_lock);
    return NULL;
}

/* caller holds g_tm_lock */
static int kot_timer_thread_locked(void)
{
    if (g_tm_started)
        return 0;
    g_tm_exit = 0;
    if (pthread_create(&g_tm_thread, NULL, kot_timer_main, NULL) != 0)
        return -1;
    g_tm_started = 1;
    return 0;
}

static void kot_timer_arm_locked(struct nv_nano_timer *t, uint64_t rel_ns)
{
    t->gen++;
    t->deadline = kot_now_ns() + rel_ns;
    t->armed = 1;
    if (kot_timer_thread_locked() != 0)
        t->armed = 0;
    pthread_cond_broadcast(&g_tm_cv);
}

/* On the worker. */
static void kot_timer_fire(void *arg)
{
    struct kot_fire *f = arg;
    struct nv_nano_timer *t = f->t;
    uint64_t gen = f->gen;
    free(f);
    pthread_mutex_lock(&g_tm_lock);
    if (t->magic != KOT_TIMER_MAGIC || t->destroyed || t->gen != gen || kot_lifecycle_fatal_entered()) {
        g_tm_dropped++;
        pthread_mutex_unlock(&g_tm_lock);
        return;
    }
    t->running = 1;
    pthread_mutex_unlock(&g_tm_lock);

    NV_STATUS st;
    if (t->kind == KOT_TIMER_RC)
        st = t->nv->rc_timer_enabled ? rm_run_rc_callback(NULL, t->nv) : NV_ERR_INVALID_STATE;
    else
        st = rm_run_nano_timer_callback(NULL, t->nv, t->pTmrEvent);

    pthread_mutex_lock(&g_tm_lock);
    t->running = 0;
    t->fired++;
    g_tm_fired++;
    /* nv.c: re-arm 1 s later while the callback succeeds and the timer is on;
     * a cancel/stop during the callback bumped gen, so it is not re-armed. */
    if (t->kind == KOT_TIMER_RC && st == NV_OK && t->nv->rc_timer_enabled && !t->destroyed && t->gen == gen)
        kot_timer_arm_locked(t, KOT_RC_PERIOD_NS);
    pthread_cond_broadcast(&g_tm_done);
    pthread_mutex_unlock(&g_tm_lock);
}

static struct nv_nano_timer *kot_timer_new(int kind, nv_state_t *nv, void *ev)
{
    struct nv_nano_timer *t = calloc(1, sizeof(*t));
    if (t == NULL)
        return NULL;
    t->magic = KOT_TIMER_MAGIC;
    t->kind = kind;
    t->nv = nv;
    t->pTmrEvent = ev;
    pthread_mutex_lock(&g_tm_lock);
    t->next = g_timers;
    g_timers = t;
    pthread_mutex_unlock(&g_tm_lock);
    return t;
}

static int kot_timer_valid(struct nv_nano_timer *t) { return t != NULL && t->magic == KOT_TIMER_MAGIC && !t->destroyed; }

/* Cancel; waits for a running callback unless called on the worker. */
static void kot_timer_cancel_sync(struct nv_nano_timer *t)
{
    int on_worker = kot_work_on_worker();
    pthread_mutex_lock(&g_tm_lock);
    t->armed = 0;
    t->gen++;
    while (t->running && !on_worker)
        pthread_cond_wait(&g_tm_done, &g_tm_lock);
    pthread_cond_broadcast(&g_tm_cv);
    pthread_mutex_unlock(&g_tm_lock);
}

void NV_API_CALL nv_create_nano_timer(nv_state_t *nv, void *pTmrEvent, nv_nano_timer_t **pnv_nstimer)
{
    if (pnv_nstimer == NULL)
        return;
    *pnv_nstimer = kot_timer_new(KOT_TIMER_NANO, nv, pTmrEvent);
    if (*pnv_nstimer == NULL)
        nv_printf(NV_DBG_ERRORS, "NVRM: Not able to create timer object \n");
}

void NV_API_CALL nv_start_nano_timer(nv_state_t *nv, nv_nano_timer_t *t, NvU64 timens)
{
    (void)nv;
    if (!kot_timer_valid(t))
        return;
    pthread_mutex_lock(&g_tm_lock);
    kot_timer_arm_locked(t, timens);
    pthread_mutex_unlock(&g_tm_lock);
}

void NV_API_CALL nv_cancel_nano_timer(nv_state_t *nv, nv_nano_timer_t *t)
{
    (void)nv;
    if (kot_timer_valid(t))
        kot_timer_cancel_sync(t);
}

void NV_API_CALL nv_destroy_nano_timer(nv_state_t *nv, nv_nano_timer_t *t)
{
    (void)nv;
    if (!kot_timer_valid(t))
        return;
    kot_timer_cancel_sync(t);
    pthread_mutex_lock(&g_tm_lock);
    t->destroyed = 1;
    pthread_mutex_unlock(&g_tm_lock);
}

/* RC timer: one per nv_state_t, found by nv. */
static struct nv_nano_timer *kot_rc_timer_for(nv_state_t *nv, int create)
{
    pthread_mutex_lock(&g_tm_lock);
    struct nv_nano_timer *t = g_timers;
    while (t != NULL && !(t->kind == KOT_TIMER_RC && t->nv == nv && !t->destroyed))
        t = t->next;
    pthread_mutex_unlock(&g_tm_lock);
    if (t == NULL && create)
        t = kot_timer_new(KOT_TIMER_RC, nv, NULL);
    return t;
}

NvS32 NV_API_CALL nv_start_rc_timer(nv_state_t *nv)
{
    if (nv == NULL || nv->rc_timer_enabled)
        return -1;
    struct nv_nano_timer *t = kot_rc_timer_for(nv, 1);
    if (t == NULL)
        return -1;
    nv_printf(NV_DBG_INFO, "NVRM: initializing rc timer\n");
    nv->rc_timer_enabled = 1;
    pthread_mutex_lock(&g_tm_lock);
    kot_timer_arm_locked(t, KOT_RC_PERIOD_NS);   /* 1 s in the future, as nv.c */
    pthread_mutex_unlock(&g_tm_lock);
    return 0;
}

NvS32 NV_API_CALL nv_stop_rc_timer(nv_state_t *nv)
{
    if (nv == NULL || !nv->rc_timer_enabled)
        return -1;
    nv_printf(NV_DBG_INFO, "NVRM: stopping rc timer\n");
    nv->rc_timer_enabled = 0;
    struct nv_nano_timer *t = kot_rc_timer_for(nv, 0);
    if (t != NULL)
        kot_timer_cancel_sync(t);
    return 0;
}

uint64_t kot_timer_fired_count(void)
{
    pthread_mutex_lock(&g_tm_lock);
    uint64_t n = g_tm_fired;
    pthread_mutex_unlock(&g_tm_lock);
    return n;
}

void kot_timer_shutdown(void)
{
    pthread_mutex_lock(&g_tm_lock);
    for (struct nv_nano_timer *t = g_timers; t != NULL; t = t->next) {
        t->armed = 0;
        t->gen++;
    }
    int started = g_tm_started;
    g_tm_exit = 1;
    pthread_cond_broadcast(&g_tm_cv);
    pthread_mutex_unlock(&g_tm_lock);
    if (started)
        pthread_join(g_tm_thread, NULL);
    pthread_mutex_lock(&g_tm_lock);
    g_tm_started = 0;
    pthread_mutex_unlock(&g_tm_lock);
}

/* Host tests only (tests/test_macos_os.c has no RM headers): a zeroed
 * nv_state_t and its rc_timer_enabled field. */
void *kot_test_nv_state_new(void) { return calloc(1, sizeof(nv_state_t)); }
int kot_test_nv_rc_timer_enabled(void *nv) { return nv ? (int)((nv_state_t *)nv)->rc_timer_enabled : -1; }
