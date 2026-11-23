/* SPDX-License-Identifier: MIT */
/*
 * shims/macos_os/os_interface_macos.c -- macOS userspace replacement for the
 * os_* services of kernel-open/nvidia/os-interface.c (OGK 570.190) that the
 * RM closure references. Prototypes come from the RM-side os-interface.h.
 * See macos_os_priv.h for the device-access rules and COVERAGE.md for status.
 */
#include <errno.h>
#include <execinfo.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvstatus.h"
#include "os-interface.h"
#include "nv.h"

#include "macos_os_priv.h"
#include "kot_lifecycle.h"

/* ======================================================================== */
/* refusal note                                                              */
/* ======================================================================== */
static pthread_mutex_t g_note_lock = PTHREAD_MUTEX_INITIALIZER;

void kot_os_note_refused(const char *fn, int *once)
{
    pthread_mutex_lock(&g_note_lock);
    int first = (*once == 0);
    *once = 1;
    pthread_mutex_unlock(&g_note_lock);
    if (first) {
        fprintf(stderr, "kot-macos-os: %s refused (not supported by the macOS OS layer)\n", fn);
    }
}

/* ======================================================================== */
/* globals the Linux file defines                                           */
/* ======================================================================== */
NvU32  os_page_size  = KOT_OS_PAGE_SIZE;
NvU64  os_page_mask  = ~((NvU64)KOT_OS_PAGE_SIZE - 1);
NvU8   os_page_shift = KOT_OS_PAGE_SHIFT;
NvBool os_cc_enabled = NV_FALSE;
NvBool os_cc_sev_snp_enabled = NV_FALSE;
NvBool os_cc_snp_vtom_enabled = NV_FALSE;
NvBool os_cc_tdx_enabled = NV_FALSE;
NvBool os_dma_buf_enabled = NV_FALSE;
NvBool os_imex_channel_is_supported = NV_FALSE;

/* Same default and filter as the Linux file: print if level >= (cur >> 4) & 3. */
static NvU32 cur_debuglevel = 0xffffffff;

/* ======================================================================== */
/* ISR context                                                               */
/* ======================================================================== */
static __thread int t_in_isr;
int kot_os_in_isr(void) { return t_in_isr; }
void kot_os_set_in_isr(int on) { t_in_isr = on ? 1 : 0; }

NvBool NV_API_CALL os_is_isr(void) { return t_in_isr ? NV_TRUE : NV_FALSE; }
NvBool NV_API_CALL os_semaphore_may_sleep(void) { return t_in_isr ? NV_FALSE : NV_TRUE; }

/* ======================================================================== */
/* memory and string helpers                                                 */
/* ======================================================================== */
NV_STATUS NV_API_CALL os_alloc_mem(void **address, NvU64 size)
{
    if (address == NULL)
        return NV_ERR_INVALID_ARGUMENT;
    *address = NULL;
    if ((NvU64)(size_t)size != size)
        return NV_ERR_INVALID_PARAMETER;
    /* kmalloc/vmalloc semantics: not zeroed; size 0 gives a unique pointer. */
    *address = malloc(size ? (size_t)size : 1);
    return (*address != NULL) ? NV_OK : NV_ERR_NO_MEMORY;
}

void NV_API_CALL os_free_mem(void *address) { free(address); }

void *NV_API_CALL os_mem_copy(void *dst, const void *src, NvU32 length)
{
    return memcpy(dst, src, length);
}

void *NV_API_CALL os_mem_set(void *dst, NvU8 c, NvU32 length)
{
    return memset(dst, c, length);
}

NvS32 NV_API_CALL os_mem_cmp(const NvU8 *buf0, const NvU8 *buf1, NvU32 length)
{
    return memcmp(buf0, buf1, length);
}

/* Single address space: "user" buffers are ordinary host memory. */
NV_STATUS NV_API_CALL os_memcpy_from_user(void *to, const void *from, NvU32 n)
{
    if (n && (to == NULL || from == NULL))
        return NV_ERR_INVALID_ADDRESS;
    memcpy(to, from, n);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_memcpy_to_user(void *to, const void *from, NvU32 n)
{
    if (n && (to == NULL || from == NULL))
        return NV_ERR_INVALID_ADDRESS;
    memcpy(to, from, n);
    return NV_OK;
}

NvS32 NV_API_CALL os_string_compare(const char *str1, const char *str2) { return strcmp(str1, str2); }
char *NV_API_CALL os_string_copy(char *dst, const char *src) { return strcpy(dst, src); }
NvU32 NV_API_CALL os_string_length(const char *str) { return (NvU32)strlen(str); }

NvU32 NV_API_CALL os_strtoul(const char *str, char **endp, NvU32 base)
{
    return (NvU32)strtoul(str, endp, (int)base);
}

NvS32 NV_API_CALL os_snprintf(char *buf, NvU32 size, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, size, fmt, ap);
    va_end(ap);
    return n;
}

/* ======================================================================== */
/* time and delays (real sleeps)                                             */
/* ======================================================================== */
NV_STATUS NV_API_CALL os_get_current_time(NvU32 *seconds, NvU32 *useconds)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *seconds = (NvU32)ts.tv_sec;
    *useconds = (NvU32)(ts.tv_nsec / 1000);
    return NV_OK;
}

NvU64 NV_API_CALL os_get_current_tick(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

NvU64 NV_API_CALL os_get_current_tick_hr(void)
{
    return (NvU64)clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
}

NvU64 NV_API_CALL os_get_tick_resolution(void)
{
    struct timespec res;
    if (clock_getres(CLOCK_MONOTONIC, &res) != 0 || (res.tv_sec == 0 && res.tv_nsec == 0))
        return 1000ull;
    return (NvU64)res.tv_sec * 1000000000ull + (NvU64)res.tv_nsec;
}

static NV_STATUS kot_sleep_ns(NvU64 ns)
{
    struct timespec req = { (time_t)(ns / 1000000000ull), (long)(ns % 1000000000ull) };
    struct timespec rem;
    while (nanosleep(&req, &rem) != 0) {
        if (errno != EINTR)
            return NV_ERR_OPERATING_SYSTEM;
        req = rem;
    }
    return NV_OK;
}

NV_STATUS NV_API_CALL os_delay_us(NvU32 MicroSeconds) { return kot_sleep_ns((NvU64)MicroSeconds * 1000ull); }
NV_STATUS NV_API_CALL os_delay(NvU32 MilliSeconds) { return kot_sleep_ns((NvU64)MilliSeconds * 1000000ull); }

NV_STATUS NV_API_CALL os_schedule(void)
{
    if (t_in_isr)
        return NV_ERR_ILLEGAL_ACTION;
    sched_yield();
    return NV_OK;
}

/* ======================================================================== */
/* process / thread / cpu                                                    */
/* ======================================================================== */
NvU32 NV_API_CALL os_get_current_process(void) { return (NvU32)getpid(); }

void NV_API_CALL os_get_current_process_name(char *buf, NvU32 len)
{
    if (buf == NULL || len == 0)
        return;
    snprintf(buf, len, "%s", getprogname());
}

NV_STATUS NV_API_CALL os_get_current_thread(NvU64 *threadId)
{
    uint64_t tid = 0;
    pthread_threadid_np(NULL, &tid);
    *threadId = tid;
    return NV_OK;
}

NV_STATUS NV_API_CALL os_get_euid(NvU32 *pSecToken)
{
    *pSecToken = (NvU32)geteuid();
    return NV_OK;
}

NvBool NV_API_CALL os_is_administrator(void) { return geteuid() == 0 ? NV_TRUE : NV_FALSE; }

NvU32 NV_API_CALL os_get_cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (NvU32)n : 1u;
}

/* macOS has no getcpu(); 0 is a valid index but not the true CPU (stub). */
NvU32 NV_API_CALL os_get_cpu_number(void) { return 0; }

/* Linux returns 0 without CONFIG_CPU_FREQ on arm64; Apple silicon exposes none. */
NvU64 NV_API_CALL os_get_cpu_frequency(void) { return 0; }

NvU64 NV_API_CALL os_get_max_user_va(void) { return 1ull << 47; }

NV_STATUS NV_API_CALL os_get_is_openrm(NvBool *bIsOpenRm)
{
    *bIsOpenRm = NV_TRUE;
    return NV_OK;
}

NV_STATUS NV_API_CALL os_get_version_info(os_version_info *pOsVersionInfo)
{
    static struct utsname u;
    if (uname(&u) != 0)
        return NV_ERR_OPERATING_SYSTEM;
    unsigned major = 0, minor = 0, build = 0;
    sscanf(u.release, "%u.%u.%u", &major, &minor, &build);
    pOsVersionInfo->os_major_version = major;
    pOsVersionInfo->os_minor_version = minor;
    pOsVersionInfo->os_build_number = build;
    pOsVersionInfo->os_build_version_str = u.version;
    pOsVersionInfo->os_build_date_plus_str = u.sysname;
    return NV_OK;
}

NV_STATUS NV_API_CALL os_get_random_bytes(NvU8 *bytes, NvU16 numBytes)
{
    arc4random_buf(bytes, numBytes);
    return NV_OK;
}

NvBool NV_API_CALL os_is_init_ns(void) { return NV_TRUE; }
NvBool NV_API_CALL os_is_nvswitch_present(void) { return NV_FALSE; }
NV_STATUS NV_API_CALL os_device_vm_present(void) { return NV_ERR_NOT_SUPPORTED; /* bare metal */ }
NvS32 NV_API_CALL os_imex_channel_count(void) { return 0; /* IMEX unsupported */ }

/* ======================================================================== */
/* logging                                                                   */
/* ======================================================================== */
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;

int NV_API_CALL nv_printf(NvU32 debuglevel, const char *printf_format, ...)
{
    int n = 0;
    if (debuglevel >= ((cur_debuglevel >> 4) & 0x3)) {
        va_list ap;
        va_start(ap, printf_format);
        pthread_mutex_lock(&g_log_lock);
        n = vfprintf(stderr, printf_format, ap);
        pthread_mutex_unlock(&g_log_lock);
        va_end(ap);
    }
    return n;
}

void NV_API_CALL out_string(const char *str)
{
    pthread_mutex_lock(&g_log_lock);
    fputs(str, stderr);
    pthread_mutex_unlock(&g_log_lock);
}

void NV_API_CALL os_dbg_init(void) { }
void NV_API_CALL os_dbg_set_level(NvU32 new_debuglevel) { cur_debuglevel = new_debuglevel; }
/* Linux: a no-op unless NVreg_EnableDbgBreakpoint is set (it is not here). */
void NV_API_CALL os_dbg_breakpoint(void) { }

void NV_API_CALL os_dump_stack(void)
{
    void *frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, 2);
}

/* A bug check parks (kot_fatal) instead of aborting while device state may be
 * owned by this process. */
void NV_API_CALL os_bug_check(NvU32 bugCode, const char *bugCodeStr)
{
    fprintf(stderr, "kot-macos-os: os_bug_check 0x%x: %s\n", bugCode, bugCodeStr ? bugCodeStr : "");
    os_dump_stack();
    kot_fatal("os_bug_check");
}

/* ======================================================================== */
/* locks (pthread)                                                           */
/* ======================================================================== */
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t c;
    NvS64 count;
} kot_sema_t;

static kot_sema_t *kot_sema_new(NvS64 initial)
{
    kot_sema_t *s = calloc(1, sizeof(*s));
    if (s == NULL)
        return NULL;
    pthread_mutex_init(&s->m, NULL);
    pthread_cond_init(&s->c, NULL);
    s->count = initial;
    return s;
}

static void kot_sema_free(kot_sema_t *s)
{
    if (s == NULL)
        return;
    pthread_cond_destroy(&s->c);
    pthread_mutex_destroy(&s->m);
    free(s);
}

static void kot_sema_down(kot_sema_t *s)
{
    pthread_mutex_lock(&s->m);
    while (s->count <= 0)
        pthread_cond_wait(&s->c, &s->m);
    s->count--;
    pthread_mutex_unlock(&s->m);
}

static int kot_sema_trydown(kot_sema_t *s)
{
    int ok = 0;
    pthread_mutex_lock(&s->m);
    if (s->count > 0) {
        s->count--;
        ok = 1;
    }
    pthread_mutex_unlock(&s->m);
    return ok;
}

static void kot_sema_up(kot_sema_t *s)
{
    pthread_mutex_lock(&s->m);
    s->count++;
    pthread_cond_signal(&s->c);
    pthread_mutex_unlock(&s->m);
}

/* mutex: Linux uses a binary semaphore (NV_INIT_MUTEX) */
NV_STATUS NV_API_CALL os_alloc_mutex(void **ppMutex)
{
    *ppMutex = kot_sema_new(1);
    return *ppMutex ? NV_OK : NV_ERR_NO_MEMORY;
}
void NV_API_CALL os_free_mutex(void *pMutex) { kot_sema_free(pMutex); }
NV_STATUS NV_API_CALL os_acquire_mutex(void *pMutex)
{
    if (t_in_isr)
        return NV_ERR_INVALID_REQUEST;
    kot_sema_down(pMutex);
    return NV_OK;
}
NV_STATUS NV_API_CALL os_cond_acquire_mutex(void *pMutex)
{
    return kot_sema_trydown(pMutex) ? NV_OK : NV_ERR_TIMEOUT_RETRY;
}
void NV_API_CALL os_release_mutex(void *pMutex) { kot_sema_up(pMutex); }

/* counting semaphore */
void *NV_API_CALL os_alloc_semaphore(NvU32 initialValue) { return kot_sema_new(initialValue); }
void NV_API_CALL os_free_semaphore(void *pSema) { kot_sema_free(pSema); }
NV_STATUS NV_API_CALL os_acquire_semaphore(void *pSema)
{
    if (t_in_isr)
        return NV_ERR_INVALID_REQUEST;
    kot_sema_down(pSema);
    return NV_OK;
}
NV_STATUS NV_API_CALL os_cond_acquire_semaphore(void *pSema)
{
    return kot_sema_trydown(pSema) ? NV_OK : NV_ERR_TIMEOUT_RETRY;
}
NV_STATUS NV_API_CALL os_release_semaphore(void *pSema)
{
    kot_sema_up(pSema);
    return NV_OK;
}

/* rwlock */
void *NV_API_CALL os_alloc_rwlock(void)
{
    pthread_rwlock_t *l = malloc(sizeof(*l));
    if (l != NULL && pthread_rwlock_init(l, NULL) != 0) {
        free(l);
        l = NULL;
    }
    return l;
}
void NV_API_CALL os_free_rwlock(void *pRwLock)
{
    if (pRwLock) {
        pthread_rwlock_destroy(pRwLock);
        free(pRwLock);
    }
}
NV_STATUS NV_API_CALL os_acquire_rwlock_read(void *pRwLock)
{
    if (t_in_isr)
        return NV_ERR_INVALID_REQUEST;
    return pthread_rwlock_rdlock(pRwLock) == 0 ? NV_OK : NV_ERR_OPERATING_SYSTEM;
}
NV_STATUS NV_API_CALL os_acquire_rwlock_write(void *pRwLock)
{
    if (t_in_isr)
        return NV_ERR_INVALID_REQUEST;
    return pthread_rwlock_wrlock(pRwLock) == 0 ? NV_OK : NV_ERR_OPERATING_SYSTEM;
}
NV_STATUS NV_API_CALL os_cond_acquire_rwlock_read(void *pRwLock)
{
    return pthread_rwlock_tryrdlock(pRwLock) == 0 ? NV_OK : NV_ERR_TIMEOUT_RETRY;
}
NV_STATUS NV_API_CALL os_cond_acquire_rwlock_write(void *pRwLock)
{
    return pthread_rwlock_trywrlock(pRwLock) == 0 ? NV_OK : NV_ERR_TIMEOUT_RETRY;
}
void NV_API_CALL os_release_rwlock_read(void *pRwLock) { pthread_rwlock_unlock(pRwLock); }
void NV_API_CALL os_release_rwlock_write(void *pRwLock) { pthread_rwlock_unlock(pRwLock); }

/*
 * Interrupt-exclusion gate (the model of Linux spin_lock_irqsave): a thread
 * holding any os_acquire_spinlock() lock is "interrupts off". The polled ISR
 * (kot_poll_interrupts, the optional poll thread) enters the gate exclusively:
 * it waits until no thread outside the ISR holds a spinlock, and while it runs
 * new spinlock acquirers wait. Nesting is counted per thread, so a thread
 * holding several spinlocks counts once; spinlocks taken inside the ISR do not
 * use the gate. The ISR does not block new acquirers while it waits (no writer
 * preference), so a spinlock holder can never deadlock behind a waiting ISR; a
 * busy system can instead make one poll give up (kot_irq_gate_enter_isr
 * returns 0) and the next poll tries again.
 */
static pthread_mutex_t g_irq_gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_irq_gate_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_irq_holders;      /* threads (outside the ISR) holding >= 1 spinlock */
static int g_irq_isr_active;
static __thread unsigned t_spin_depth;

static void kot_irq_gate_hold(void)
{
    if (t_in_isr || t_spin_depth++ != 0)
        return;
    pthread_mutex_lock(&g_irq_gate);
    while (g_irq_isr_active)
        pthread_cond_wait(&g_irq_gate_cv, &g_irq_gate);
    g_irq_holders++;
    pthread_mutex_unlock(&g_irq_gate);
}

static void kot_irq_gate_unhold(void)
{
    if (t_in_isr || t_spin_depth == 0 || --t_spin_depth != 0)
        return;
    pthread_mutex_lock(&g_irq_gate);
    g_irq_holders--;
    pthread_cond_broadcast(&g_irq_gate_cv);
    pthread_mutex_unlock(&g_irq_gate);
}

int kot_irq_gate_enter_isr(uint64_t timeout_ns)
{
    if (t_spin_depth != 0)
        return 0;   /* this thread holds a spinlock: interrupts are "off" here */
    pthread_mutex_lock(&g_irq_gate);
    uint64_t deadline = os_get_current_tick() + timeout_ns;
    while (g_irq_holders != 0 || g_irq_isr_active) {
        uint64_t now = os_get_current_tick();
        if (now >= deadline) {
            pthread_mutex_unlock(&g_irq_gate);
            return 0;
        }
        struct timespec rel = { (time_t)((deadline - now) / 1000000000ull), (long)((deadline - now) % 1000000000ull) };
        pthread_cond_timedwait_relative_np(&g_irq_gate_cv, &g_irq_gate, &rel);
    }
    g_irq_isr_active = 1;
    pthread_mutex_unlock(&g_irq_gate);
    t_in_isr = 1;
    return 1;
}

void kot_irq_gate_exit_isr(void)
{
    t_in_isr = 0;
    pthread_mutex_lock(&g_irq_gate);
    g_irq_isr_active = 0;
    pthread_cond_broadcast(&g_irq_gate_cv);
    pthread_mutex_unlock(&g_irq_gate);
}

/* spinlock: a pthread mutex inside the interrupt gate; "old irql" is always 0 */
NV_STATUS NV_API_CALL os_alloc_spinlock(void **ppSpinlock)
{
    pthread_mutex_t *m = malloc(sizeof(*m));
    if (m == NULL)
        return NV_ERR_NO_MEMORY;
    pthread_mutex_init(m, NULL);
    *ppSpinlock = m;
    return NV_OK;
}
void NV_API_CALL os_free_spinlock(void *pSpinlock)
{
    if (pSpinlock) {
        pthread_mutex_destroy(pSpinlock);
        free(pSpinlock);
    }
}
NvU64 NV_API_CALL os_acquire_spinlock(void *pSpinlock)
{
    kot_irq_gate_hold();
    pthread_mutex_lock(pSpinlock);
    return 0;
}
void NV_API_CALL os_release_spinlock(void *pSpinlock, NvU64 oldIrql)
{
    (void)oldIrql;
    pthread_mutex_unlock(pSpinlock);
    kot_irq_gate_unhold();
}

/* wait queue: a completion (count of wake-ups) */
struct os_wait_queue {
    pthread_mutex_t m;
    pthread_cond_t c;
    NvU64 done;
};

NV_STATUS NV_API_CALL os_alloc_wait_queue(os_wait_queue **wq)
{
    os_wait_queue *q = calloc(1, sizeof(*q));
    if (q == NULL)
        return NV_ERR_NO_MEMORY;
    pthread_mutex_init(&q->m, NULL);
    pthread_cond_init(&q->c, NULL);
    *wq = q;
    return NV_OK;
}
void NV_API_CALL os_free_wait_queue(os_wait_queue *wq)
{
    if (wq) {
        pthread_cond_destroy(&wq->c);
        pthread_mutex_destroy(&wq->m);
        free(wq);
    }
}
static void kot_wq_wait(os_wait_queue *wq)
{
    pthread_mutex_lock(&wq->m);
    while (wq->done == 0)
        pthread_cond_wait(&wq->c, &wq->m);
    wq->done--;
    pthread_mutex_unlock(&wq->m);
}
void NV_API_CALL os_wait_uninterruptible(os_wait_queue *wq) { kot_wq_wait(wq); }
void NV_API_CALL os_wait_interruptible(os_wait_queue *wq) { kot_wq_wait(wq); }
void NV_API_CALL os_wake_up(os_wait_queue *wq)
{
    pthread_mutex_lock(&wq->m);
    wq->done++;
    pthread_cond_signal(&wq->c);
    pthread_mutex_unlock(&wq->m);
}

/* ======================================================================== */
/* work queue: one dedicated worker thread, FIFO                            */
/* ======================================================================== */
/*
 * Every os_work_queue maps to one worker thread. RM items run as
 * rm_execute_work_item(NULL, data): on arm64 NV_ENTER_RM_RUNTIME is
 * "{ (void)sp; (void)fp; }" (osapi.h; the altstack form exists only for
 * NVCPU_X86_64 with __use_altstack__), so RM never dereferences the stack
 * pointer and NULL is what the 570.190 source needs. The layer's own deferred
 * callbacks (nano and RC timer expiries, kot_timer_macos.c) run on the same
 * thread through kot_work_post(), so RM sees them in the same context its
 * Linux work-queue callbacks would run in.
 * os_flush_work_queue() waits for the items queued before it was called (not
 * for items queued later, e.g. by a periodic timer). kot_work_queue_stop()
 * refuses new items, drains the queue and ends the worker; it is called at
 * teardown. After kot_fatal() the worker executes nothing more.
 */
typedef struct kot_work {
    void (*fn)(void *);   /* NULL: RM item, rm_execute_work_item(NULL, data) */
    void *data;
    uint64_t seq;
    struct kot_work *next;
} kot_work_t;

static pthread_mutex_t g_wq_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_wq_cond = PTHREAD_COND_INITIALIZER;   /* new work / stop */
static pthread_cond_t g_wq_done = PTHREAD_COND_INITIALIZER;   /* an item completed */
static kot_work_t *g_wq_head, *g_wq_tail;
static int g_wq_started;
static int g_wq_stopping;          /* kot_work_queue_stop: refuse new items */
static int g_wq_exit;              /* worker leaves once the queue is empty */
static NvBool g_wq_unload_flush;
static uint64_t g_wq_executed;
static uint64_t g_wq_queued_seq;   /* seq of the last item queued */
static uint64_t g_wq_done_seq;     /* seq of the last item completed (FIFO) */
static uint64_t g_wq_refused;
static pthread_t g_wq_thread;

static void *kot_worker(void *arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_wq_lock);
        while (g_wq_head == NULL && !g_wq_exit)
            pthread_cond_wait(&g_wq_cond, &g_wq_lock);
        if (g_wq_head == NULL) {
            pthread_mutex_unlock(&g_wq_lock);
            return NULL;
        }
        kot_work_t *w = g_wq_head;
        g_wq_head = w->next;
        if (g_wq_head == NULL)
            g_wq_tail = NULL;
        pthread_mutex_unlock(&g_wq_lock);

        if (!kot_lifecycle_fatal_entered()) {
            if (w->fn != NULL)
                w->fn(w->data);
            else
                rm_execute_work_item(NULL, w->data);
        }

        pthread_mutex_lock(&g_wq_lock);
        g_wq_executed++;
        g_wq_done_seq = w->seq;
        pthread_cond_broadcast(&g_wq_done);
        pthread_mutex_unlock(&g_wq_lock);
        free(w);
    }
}

uint64_t kot_work_executed_count(void)
{
    pthread_mutex_lock(&g_wq_lock);
    uint64_t n = g_wq_executed;
    pthread_mutex_unlock(&g_wq_lock);
    return n;
}

int kot_work_on_worker(void)
{
    pthread_mutex_lock(&g_wq_lock);
    int on = g_wq_started && pthread_equal(pthread_self(), g_wq_thread);
    pthread_mutex_unlock(&g_wq_lock);
    return on;
}

static NV_STATUS kot_work_enqueue(void (*fn)(void *), void *data)
{
    kot_work_t *w = malloc(sizeof(*w));
    if (w == NULL)
        return NV_ERR_NO_MEMORY;
    w->fn = fn;
    w->data = data;
    w->next = NULL;
    pthread_mutex_lock(&g_wq_lock);
    if (g_wq_stopping || kot_lifecycle_fatal_entered()) {
        g_wq_refused++;
        pthread_mutex_unlock(&g_wq_lock);
        free(w);
        return NV_ERR_NOT_READY;
    }
    if (!g_wq_started) {
        if (pthread_create(&g_wq_thread, NULL, kot_worker, NULL) != 0) {
            pthread_mutex_unlock(&g_wq_lock);
            free(w);
            return NV_ERR_NOT_READY;
        }
        g_wq_started = 1;
    }
    w->seq = ++g_wq_queued_seq;
    if (g_wq_tail)
        g_wq_tail->next = w;
    else
        g_wq_head = w;
    g_wq_tail = w;
    pthread_cond_signal(&g_wq_cond);
    pthread_mutex_unlock(&g_wq_lock);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_queue_work_item(struct os_work_queue *queue, void *data)
{
    (void)queue;   /* every queue maps to the single worker */
    return kot_work_enqueue(NULL, data);
}

int kot_work_post(void (*fn)(void *), void *arg)
{
    return fn != NULL && kot_work_enqueue(fn, arg) == NV_OK ? 0 : -1;
}

NV_STATUS NV_API_CALL os_flush_work_queue(struct os_work_queue *queue, NvBool is_unload)
{
    (void)queue;
    if (t_in_isr)
        return NV_ERR_ILLEGAL_ACTION;
    pthread_mutex_lock(&g_wq_lock);
    if (g_wq_started && pthread_equal(pthread_self(), g_wq_thread)) {
        pthread_mutex_unlock(&g_wq_lock);
        return NV_ERR_ILLEGAL_ACTION;   /* flushing from the worker would deadlock */
    }
    uint64_t target = g_wq_queued_seq;   /* only what was queued before this call */
    g_wq_unload_flush = is_unload;
    while (g_wq_started && g_wq_done_seq < target)
        pthread_cond_wait(&g_wq_done, &g_wq_lock);
    g_wq_unload_flush = NV_FALSE;
    pthread_mutex_unlock(&g_wq_lock);
    return NV_OK;
}

int kot_work_queue_stop(void)
{
    pthread_mutex_lock(&g_wq_lock);
    if (g_wq_started && pthread_equal(pthread_self(), g_wq_thread)) {
        pthread_mutex_unlock(&g_wq_lock);
        return -1;
    }
    g_wq_stopping = 1;
    int started = g_wq_started;
    if (started) {
        g_wq_exit = 1;
        pthread_cond_broadcast(&g_wq_cond);
    }
    uint64_t refused = g_wq_refused;
    pthread_mutex_unlock(&g_wq_lock);
    if (started)
        pthread_join(g_wq_thread, NULL);
    pthread_mutex_lock(&g_wq_lock);
    g_wq_started = 0;
    pthread_mutex_unlock(&g_wq_lock);
    return (int)refused;
}

uint64_t kot_work_refused_count(void)
{
    pthread_mutex_lock(&g_wq_lock);
    uint64_t n = g_wq_refused;
    pthread_mutex_unlock(&g_wq_lock);
    return n;
}

NvBool NV_API_CALL os_is_queue_flush_ongoing(struct os_work_queue *queue)
{
    (void)queue;
    pthread_mutex_lock(&g_wq_lock);
    NvBool b = g_wq_unload_flush;
    pthread_mutex_unlock(&g_wq_lock);
    return b;
}

/* ======================================================================== */
/* BAR mapping: the shim_io.c token scheme                                   */
/* ======================================================================== */
/*
 * Returns a registry token (shims/macos_os/kot_tokens.h) for [start,
 * start+size_bytes), which must lie inside one BAR of the accepted device as
 * shim_pci.c / kot_device.h present it; anything else returns NULL. The token
 * is not a CPU pointer: shim_read{b,w,l,q} and shim_write{b,w,l,q} validate it.
 */
void *NV_API_CALL os_map_kernel_space(NvU64 start, NvU64 size_bytes, NvU32 mode)
{
    (void)mode;
    if (size_bytes == 0 || (NvU64)(unsigned long)size_bytes != size_bytes)
        return NULL;
    return ioremap(start, (unsigned long)size_bytes);
}

void NV_API_CALL os_unmap_kernel_space(void *addr, NvU64 size_bytes)
{
    (void)size_bytes;
    iounmap(addr);
}

/* ======================================================================== */
/* cache maintenance                                                         */
/* ======================================================================== */
void NV_API_CALL os_flush_cpu_write_combine_buffer(void)
{
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __builtin_arm_dsb(0xf);   /* dsb sy, as Linux wmb() on arm64 */
}

/* No userspace set/way cache flush on arm64 (Linux arm64 refuses too). */
NV_STATUS NV_API_CALL os_flush_cpu_cache_all(void) { KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
NV_STATUS NV_API_CALL os_flush_user_cache(void) { KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }

/* ======================================================================== */
/* refused: NUMA, pages, crash dump, ACPI/EFI, vGPU, console, files, caps,   */
/* pid namespaces, x86 I/O ports, confidential compute, IMEX                 */
/* ======================================================================== */
NV_STATUS NV_API_CALL os_alloc_pages_node(NvS32 nid, NvU32 size, NvU32 flag, NvU64 *pAddress)
{
    (void)nid; (void)size; (void)flag;
    KOT_REFUSED();
    if (pAddress) *pAddress = 0;
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_get_page(NvU64 address) { (void)address; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
NV_STATUS NV_API_CALL os_put_page(NvU64 address) { (void)address; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }
NvU32 NV_API_CALL os_get_page_refcount(NvU64 address) { (void)address; KOT_REFUSED(); return 0; }
NvU32 NV_API_CALL os_count_tail_pages(NvU64 address) { (void)address; KOT_REFUSED(); return 0; }
NV_STATUS NV_API_CALL os_offline_page_at_address(NvU64 address) { (void)address; KOT_REFUSED(); return NV_ERR_NOT_SUPPORTED; }

NV_STATUS NV_API_CALL os_numa_memblock_size(NvU64 *memblock_size)
{
    KOT_REFUSED();
    if (memblock_size) *memblock_size = 0;
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_numa_add_gpu_memory(void *handle, NvU64 offset, NvU64 size, NvU32 *nodeId)
{
    (void)handle; (void)offset; (void)size; (void)nodeId;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_numa_remove_gpu_memory(void *handle, NvU64 offset, NvU64 size, NvU32 nodeId)
{
    (void)handle; (void)offset; (void)size; (void)nodeId;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_get_numa_node_memory_usage(NvS32 node_id, NvU64 *free_memory_bytes, NvU64 *total_memory_bytes)
{
    (void)node_id; (void)free_memory_bytes; (void)total_memory_bytes;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}

void NV_API_CALL os_add_record_for_crashLog(void *pbuffer, NvU32 size) { (void)pbuffer; (void)size; KOT_REFUSED(); }
void NV_API_CALL os_delete_record_for_crashLog(void *pbuffer) { (void)pbuffer; KOT_REFUSED(); }

NV_STATUS NV_API_CALL os_get_acpi_rsdp_from_uefi(NvU32 *pRsdpAddr)
{
    KOT_REFUSED();
    if (pRsdpAddr) *pRsdpAddr = 0;
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_get_smbios_header(NvU64 *pSmbsAddr)
{
    KOT_REFUSED();
    if (pSmbsAddr) *pSmbsAddr = 0;
    return NV_ERR_NOT_SUPPORTED;
}
NvBool NV_API_CALL os_is_efi_enabled(void) { KOT_REFUSED(); return NV_FALSE; }

NvBool NV_API_CALL os_is_grid_supported(void) { return NV_FALSE; }
NvU32 NV_API_CALL os_get_grid_csp_support(void) { return 0; }
NvBool NV_API_CALL os_is_vgx_hyper(void) { return NV_FALSE; }
NV_STATUS NV_API_CALL os_call_vgpu_vfio(void *pvgpu_vfio_info, NvU32 cmd_type)
{
    (void)pvgpu_vfio_info; (void)cmd_type;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_inject_vgx_msi(NvU16 guestID, NvU64 msiAddr, NvU32 msiData)
{
    (void)guestID; (void)msiAddr; (void)msiData;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}

void NV_API_CALL os_disable_console_access(void) { KOT_REFUSED(); }
void NV_API_CALL os_enable_console_access(void) { KOT_REFUSED(); }

NV_STATUS NV_API_CALL os_open_temporary_file(void **ppFile)
{
    KOT_REFUSED();
    if (ppFile) *ppFile = NULL;
    return NV_ERR_NOT_SUPPORTED;
}
void NV_API_CALL os_close_file(void *pFile) { (void)pFile; KOT_REFUSED(); }
NV_STATUS NV_API_CALL os_write_file(void *pFile, NvU8 *pBuffer, NvU64 size, NvU64 offset)
{
    (void)pFile; (void)pBuffer; (void)size; (void)offset;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
NV_STATUS NV_API_CALL os_read_file(void *pFile, NvU8 *pBuffer, NvU64 size, NvU64 offset)
{
    (void)pFile; (void)pBuffer; (void)size; (void)offset;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}

nv_cap_t *NV_API_CALL os_nv_cap_create_dir_entry(nv_cap_t *parent_cap, const char *name, int mode)
{
    (void)parent_cap; (void)name; (void)mode;
    KOT_REFUSED();
    return NULL;
}
nv_cap_t *NV_API_CALL os_nv_cap_create_file_entry(nv_cap_t *parent_cap, const char *name, int mode)
{
    (void)parent_cap; (void)name; (void)mode;
    KOT_REFUSED();
    return NULL;
}
void NV_API_CALL os_nv_cap_destroy_entry(nv_cap_t *cap) { (void)cap; KOT_REFUSED(); }
int NV_API_CALL os_nv_cap_validate_and_dup_fd(const nv_cap_t *cap, int fd)
{
    (void)cap; (void)fd;
    KOT_REFUSED();
    return -1;
}
void NV_API_CALL os_nv_cap_close_fd(int fd) { (void)fd; KOT_REFUSED(); }

/* Capability checks are not granted on this path. */
NvBool NV_API_CALL os_check_access(RsAccessRight accessRight) { (void)accessRight; KOT_REFUSED(); return NV_FALSE; }

NV_STATUS NV_API_CALL os_find_ns_pid(void *pid_info, NvU32 *ns_pid)
{
    (void)pid_info; (void)ns_pid;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}
void *NV_API_CALL os_get_pid_info(void) { KOT_REFUSED(); return NULL; }
void NV_API_CALL os_put_pid_info(void *pid_info) { (void)pid_info; }

NvU8 NV_API_CALL os_io_read_byte(NvU32 address) { (void)address; KOT_REFUSED(); return 0xff; }
NvU16 NV_API_CALL os_io_read_word(NvU32 address) { (void)address; KOT_REFUSED(); return 0xffff; }
void NV_API_CALL os_io_write_byte(NvU32 address, NvU8 value) { (void)address; (void)value; KOT_REFUSED(); }
void NV_API_CALL os_io_write_word(NvU32 address, NvU16 value) { (void)address; (void)value; KOT_REFUSED(); }

NV_STATUS NV_API_CALL os_enable_pci_req_atomics(void *handle, enum os_pci_req_atomics_type type)
{
    (void)handle; (void)type;
    KOT_REFUSED();
    return NV_ERR_NOT_SUPPORTED;
}

NvS32 NV_API_CALL os_imex_channel_get(NvU64 descriptor) { (void)descriptor; KOT_REFUSED(); return -1; }
