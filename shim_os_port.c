#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_driverkit_kpi.h>
#include <linux/spinlock.h>
#include <shim_mutation_admission.h>

void spin_lock_init(spinlock_t *lock) { if (lock) lock->locked = 0; }
void spin_lock(spinlock_t *lock) { if (lock) lock->locked = 1; }
void spin_unlock(spinlock_t *lock) { if (lock) lock->locked = 0; }
unsigned long __shim_spin_lock_irqsave(spinlock_t *lock) { spin_lock(lock); return 0; }
void spin_unlock_irqrestore(spinlock_t *lock, unsigned long flags) { (void)flags; spin_unlock(lock); }

#if !defined(KOT_RM_CLOSURE_LINK) /* superseded by the real RM objects in the closure link */
void *portMemAllocNonPaged(uint64_t size) { return calloc(1, (size_t)size); }
void portMemFree(void *ptr) { free(ptr); }
void *portMemSet(void *dst, int value, uint64_t size) { return memset(dst, value, (size_t)size); }
void *portMemCopy(void *dst, const void *src, uint64_t size) { return memcpy(dst, src, (size_t)size); }

void nvAssertFailedNoLog(void) {}
void nvAssertOkFailedNoLog(void) {}
void nvCheckFailedNoLog(void) {}
void nvCheckOkFailedNoLog(void) {}
void nvErrorLog_va(void *gpu, const char *fmt, va_list ap) { (void)gpu; (void)fmt; (void)ap; }
void osSpinLoop(void) {}
#endif
#if !defined(KOT_RM_CLOSURE_LINK)
void udelay(unsigned long usecs) { (void)usecs; }
void mdelay(unsigned long msecs) { (void)msecs; }
void msleep(unsigned int msecs) { (void)msecs; }
void usleep_range(unsigned long min, unsigned long max) { (void)min; (void)max; }
void ndelay(unsigned long nsecs) { (void)nsecs; }
#else /* closure link: real delays, never silent no-ops */
static void kot_sleep_ns(uint64_t ns)
{
    struct timespec req = { (time_t)(ns / 1000000000ull), (long)(ns % 1000000000ull) }, rem;
    while (nanosleep(&req, &rem) != 0) req = rem;
}
void udelay(unsigned long usecs) { kot_sleep_ns((uint64_t)usecs * 1000ull); }
void mdelay(unsigned long msecs) { kot_sleep_ns((uint64_t)msecs * 1000000ull); }
void msleep(unsigned int msecs) { kot_sleep_ns((uint64_t)msecs * 1000000ull); }
void usleep_range(unsigned long min, unsigned long max) { (void)max; kot_sleep_ns((uint64_t)min * 1000ull); }
void ndelay(unsigned long nsecs) { kot_sleep_ns((uint64_t)nsecs); }
#endif
#if !defined(KOT_RM_CLOSURE_LINK)
uint32_t osReadRegistryDword(void *os, const char *name, uint32_t value) { (void)os; (void)name; return value; }
void osQueueWorkItem(void *queue, void *item) { (void)queue; (void)item; }

uint32_t regRead032(void *gpu, uint32_t offset)
{
    (void)gpu;
    uint32_t value = 0xffffffffU;
    (void)cudaDriverKitKpiReadReg32(0, offset, &value);
    return value;
}

void timeoutSet(void *timeout, uint32_t usec) { (void)timeout; (void)usec; }
int timeoutCheck(void *timeout) { (void)timeout; return 0; }

// Real OGK contract (src/nvidia/inc/kernel/gpu/gpu_timeout.h):
//   NV_STATUS timeoutCondWait(TIMEOUT_DATA*, RMTIMEOUT*, GpuWaitConditionFunc*,
//                             void *pCondData, NvU32 line)
// where GpuWaitConditionFunc = NvBool(*)(OBJGPU *pGpu, void *pVoid) and
// pGpu = ((TIMEOUT_DATA*)pTD)->pGpu (first field). The previous 2-arg stub
// dropped the condition fn and returned NV_OK immediately, so a poll like
// kfspWaitForSecureBoot_GB202 never read silicon. This drives the real condition
// (which reads NV_THERM_I2CS_SCRATCH via regRead032 → cudaDriverKitKpiReadReg32)
// in a wall-clock-bounded loop. KOT_COND_TIMEOUT_MS bounds the poll (default 200);
// KOT_FSP_PROBE=1 returns NV_OK on timeout (host-side dry run skips the OGK
// failure path); otherwise the honest NV_ERR_TIMEOUT (0x41) is returned.
int timeoutCondWait(void *pTD, void *pTimeout, void *pCondFunc, void *pCondData, unsigned int line)
{
    (void)pTimeout;
    (void)line;
    typedef unsigned char (*cond_fn)(void *pGpu, void *pData);
    cond_fn cond = (cond_fn)pCondFunc;
    if (cond == NULL) {
        return 0; /* NV_OK — no condition to test */
    }
    void *pGpu = (pTD != NULL) ? *(void **)pTD : NULL; /* TIMEOUT_DATA.pGpu @ off 0 */
    long budget_ms = 200;
    const char *b = getenv("KOT_COND_TIMEOUT_MS");
    if (b != NULL) {
        budget_ms = atol(b);
    }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        if (cond(pGpu, pCondData)) {
            return 0; /* NV_OK — condition met (e.g. FSP boot complete) */
        }
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
        if (ms >= budget_ms) {
            if (cond(pGpu, pCondData)) {
                return 0;
            }
            return (getenv("KOT_FSP_PROBE") != NULL) ? 0 : 0x41; /* NV_ERR_TIMEOUT */
        }
        usleep(1000);
    }
}
void threadStateResetTimeout(void *thread) { (void)thread; }
int tmrEventCreate_IMPL(void *tmr, void **event) { (void)tmr; if (event) *event = calloc(1, 1); return 0; }
void tmrEventDestroy_IMPL(void *tmr, void *event) { (void)tmr; free(event); }
int tmrEventScheduleRel_IMPL(void *tmr, void *event, uint64_t ns) { (void)tmr; (void)event; (void)ns; return 0; }

void *memdescCreate(void) { return calloc(1, 64); }
int memdescAlloc(void *m) { (void)m; return 0; }
void memdescFree(void *m) { (void)m; }
void memdescDestroy(void *m) { free(m); }
void *memdescMap(void *m, ...) { return m; }
void *memdescMapInternal(void *m, ...) { return m; }
void memdescUnmap(void *m, ...) { (void)m; }
void memdescUnmapInternal(void *m, ...) { (void)m; }
void *memdescGetKernelMapping(void *m) { return m; }
void *memdescGetKernelMappingPriv(void *m) { return m; }
void memdescSetKernelMapping(void *m, void *p) { (void)m; (void)p; }
void memdescSetKernelMappingPriv(void *m, void *p) { (void)m; (void)p; }
uint64_t memdescGetPhysAddrs(void *m, ...) { (void)m; return 0; }
void memdescSetFlag(void *m, uint32_t flag, uint32_t value) { (void)m; (void)flag; (void)value; }
void memdescSetPageSize(void *m, uint32_t page_size) { (void)m; (void)page_size; }

uint64_t bindataGetBufferSize(void *b) { (void)b; return 0; }
void *bindataArchiveGetStorage(void *archive, const char *name) { (void)archive; (void)name; return 0; }
int bindataStorageAcquireData(void *storage, const void **data, uint64_t *size)
{
    (void)storage;
    if (data) *data = 0;
    if (size) *size = 0;
    return -1;
}
void bindataStorageReleaseData(void *storage, const void *data) { (void)storage; (void)data; }
int bindataWriteToBuffer(void *bindata, void *buffer, uint64_t size) { (void)bindata; (void)buffer; (void)size; return -1; }
#endif /* !KOT_RM_CLOSURE_LINK */

#if !defined(KOT_RM_CLOSURE_LINK)
/* The msgq reimplementation and the typed GSP RPC shim below are replaced, in
 * the closure link, by the real src/common/shared/msgq/msgq.c and RM's own
 * RPC layer (the wire layout of msgqTxHeader/msgqRxHeader is the same). */
typedef void *msgqHandle;

#define SHIM_MSGQ_MAGIC 0x4d534751U
#define SHIM_MSGQ_VERSION 0U
#define SHIM_MSGQ_SNAPSHOT_VERSION 1U
#define SHIM_MSGQ_MSG_SIZE_MIN 16U
#define SHIM_MSGQ_META_MIN_ALIGN 3U
#define SHIM_MSGQ_META_MAX_ALIGN 20U
#define SHIM_MSGQ_FLAGS_SWAP_RX 1U
#define SHIM_MSGQ_OWNERSHIP_PLAN_VERSION 1U
#define SHIM_MSGQ_OWNERSHIP_SNAPSHOT_VERSION 1U
#define SHIM_MSGQ_OWNER_UNKNOWN 0U
#define SHIM_MSGQ_OWNER_RM 1U
#define SHIM_MSGQ_OWNER_GSP 2U
#define SHIM_MSGQ_OWNER_NATIVE_RM 3U
#define SHIM_MSGQ_QUEUE_KIND_NONE 0U
#define SHIM_MSGQ_QUEUE_KIND_MSGQ 1U
#define SHIM_MSGQ_QUEUE_KIND_NATIVE_RM 2U
#define SHIM_MSGQ_HANDOFF_NONE 0U
#define SHIM_MSGQ_HANDOFF_MSGQ_ACTIVE 1U
#define SHIM_MSGQ_HANDOFF_NATIVE_RM_PLANNED 2U

typedef struct
{
    uint32_t version;
    uint32_t size;
    uint32_t msgSize;
    uint32_t msgCount;
    uint32_t writePtr;
    uint32_t flags;
    uint32_t rxHdrOff;
    uint32_t entryOff;
} shim_msgq_tx_header;

typedef struct
{
    uint32_t readPtr;
} shim_msgq_rx_header;

typedef struct
{
    uint32_t magic;
    uint32_t init_count;

    shim_msgq_tx_header tx;
    shim_msgq_tx_header rx;

    shim_msgq_tx_header *our_tx_hdr;
    const shim_msgq_tx_header *their_tx_hdr;
    shim_msgq_rx_header *our_rx_hdr;
    shim_msgq_rx_header *their_rx_hdr;
    uint8_t *our_entries;
    const uint8_t *their_entries;

    uint32_t tx_free;
    uint32_t rx_read_ptr;
    uint32_t rx_avail;

    uint32_t tx_create_count;
    uint32_t tx_get_write_count;
    uint32_t tx_submit_count;
    uint32_t rx_link_count;
    uint32_t rx_get_read_count;
    uint32_t rx_consume_count;

    uint32_t last_tx_submit_count;
    uint32_t last_rx_consume_count;
    uint32_t last_write_index;
    uint32_t last_read_index;

    uint32_t endpoint_owner;
    uint32_t peer_owner;
    uint32_t handoff_state;
    uint32_t native_queue_owner;
    uint32_t native_queue_peer;
    uint32_t native_queue_kind;
    uint64_t native_queue_token;

    uint8_t tx_linked;
    uint8_t rx_linked;
    uint8_t rx_swapped;
    uint8_t reserved;
} shim_msgq;

typedef struct
{
    uint32_t version;
    uint32_t init_count;
    uint32_t tx_linked;
    uint32_t rx_linked;
    uint32_t rx_swapped;
    uint32_t tx_create_count;
    uint32_t tx_get_write_count;
    uint32_t tx_submit_count;
    uint32_t rx_link_count;
    uint32_t rx_get_read_count;
    uint32_t rx_consume_count;
    uint32_t last_tx_submit_count;
    uint32_t last_rx_consume_count;
    uint32_t tx_write_ptr;
    uint32_t tx_read_ptr;
    uint32_t tx_free;
    uint32_t tx_msg_count;
    uint32_t tx_msg_size;
    uint32_t tx_entry_off;
    uint32_t rx_write_ptr;
    uint32_t rx_read_ptr;
    uint32_t rx_avail;
    uint32_t rx_msg_count;
    uint32_t rx_msg_size;
    uint32_t rx_entry_off;
} shim_msgq_snapshot;

typedef struct
{
    uint32_t version;
    uint32_t endpoint_owner;
    uint32_t peer_owner;
    uint32_t native_queue_owner;
    uint32_t native_queue_peer;
    uint32_t native_queue_kind;
    uint32_t handoff_state;
    uint32_t reserved;
    uint64_t native_queue_token;
} shim_msgq_ownership_plan;

typedef struct
{
    uint32_t queue_kind;
    uint32_t producer_owner;
    uint32_t consumer_owner;
    uint32_t msg_count;
    uint32_t msg_size;
    uint32_t write_ptr;
    uint32_t read_ptr;
    uint32_t entry_off;
} shim_msgq_queue_ownership;

typedef struct
{
    uint32_t version;
    uint32_t endpoint_owner;
    uint32_t peer_owner;
    uint32_t handoff_state;
    uint32_t native_queue_owner;
    uint32_t native_queue_peer;
    uint32_t native_queue_kind;
    uint32_t reserved;
    uint64_t native_queue_token;
    shim_msgq_queue_ownership tx;
    shim_msgq_queue_ownership rx;
} shim_msgq_ownership_snapshot;

static int shim_msgq_is_valid(const shim_msgq *q)
{
    return (q != 0) && (q->magic == SHIM_MSGQ_MAGIC);
}

static int shim_msgq_owner_is_valid(uint32_t owner)
{
    return owner == SHIM_MSGQ_OWNER_UNKNOWN ||
           owner == SHIM_MSGQ_OWNER_RM ||
           owner == SHIM_MSGQ_OWNER_GSP ||
           owner == SHIM_MSGQ_OWNER_NATIVE_RM;
}

static int shim_msgq_queue_kind_is_valid(uint32_t kind)
{
    return kind == SHIM_MSGQ_QUEUE_KIND_NONE ||
           kind == SHIM_MSGQ_QUEUE_KIND_MSGQ ||
           kind == SHIM_MSGQ_QUEUE_KIND_NATIVE_RM;
}

static int shim_msgq_handoff_state_is_valid(uint32_t state)
{
    return state == SHIM_MSGQ_HANDOFF_NONE ||
           state == SHIM_MSGQ_HANDOFF_MSGQ_ACTIVE ||
           state == SHIM_MSGQ_HANDOFF_NATIVE_RM_PLANNED;
}

static uint32_t shim_msgq_default_peer_owner(uint32_t owner)
{
    if (owner == SHIM_MSGQ_OWNER_RM)
    {
        return SHIM_MSGQ_OWNER_GSP;
    }
    if (owner == SHIM_MSGQ_OWNER_GSP)
    {
        return SHIM_MSGQ_OWNER_RM;
    }
    return SHIM_MSGQ_OWNER_UNKNOWN;
}

static uint32_t shim_msgq_resolved_peer_owner(const shim_msgq *q)
{
    if (q->peer_owner != SHIM_MSGQ_OWNER_UNKNOWN)
    {
        return q->peer_owner;
    }
    return shim_msgq_default_peer_owner(q->endpoint_owner);
}

static uint32_t shim_msgq_resolved_handoff_state(const shim_msgq *q)
{
    if (q->handoff_state != SHIM_MSGQ_HANDOFF_NONE)
    {
        return q->handoff_state;
    }
    if (q->tx_linked || q->rx_linked)
    {
        return SHIM_MSGQ_HANDOFF_MSGQ_ACTIVE;
    }
    return SHIM_MSGQ_HANDOFF_NONE;
}

static uint32_t shim_msgq_align_up_u32(uint32_t value, unsigned align_shift)
{
    uint32_t align = 1U << align_shift;
    return (value + align - 1U) & ~(align - 1U);
}

static uint32_t shim_msgq_advance(uint32_t ptr, uint32_t count, uint32_t delta)
{
    if (count == 0)
    {
        return 0;
    }
    return (ptr + delta) % count;
}

unsigned msgqGetMetaSize(void)
{
    return (unsigned)sizeof(shim_msgq);
}

int msgqInit(msgqHandle *pHandle, void *pBuffer)
{
    shim_msgq *q = (shim_msgq *)pBuffer;

    if (q == 0)
    {
        return -1;
    }

    memset(q, 0, sizeof(*q));
    q->magic = SHIM_MSGQ_MAGIC;
    q->init_count = 1;

    if (pHandle != 0)
    {
        *pHandle = q;
    }

    return 0;
}

static unsigned shim_msgq_tx_free_space(shim_msgq *q)
{
    uint32_t read_ptr;
    uint32_t free_count;

    if (!shim_msgq_is_valid(q) || !q->tx_linked || q->tx.msgCount < 2)
    {
        return 0;
    }

    read_ptr = q->our_rx_hdr != 0 ? q->our_rx_hdr->readPtr : 0;
    if (read_ptr >= q->tx.msgCount)
    {
        return 0;
    }

    free_count = read_ptr + q->tx.msgCount - q->tx.writePtr - 1U;
    if (free_count >= q->tx.msgCount)
    {
        free_count -= q->tx.msgCount;
    }

    q->tx_free = free_count;
    return free_count;
}

unsigned msgqTxGetFreeSpace(msgqHandle handle)
{
    return shim_msgq_tx_free_space((shim_msgq *)handle);
}

int msgqTxCreate(msgqHandle handle, void *pBackingStore, unsigned size,
                 unsigned msgSize, unsigned hdrAlign, unsigned entryAlign,
                 unsigned flags)
{
    shim_msgq *q = (shim_msgq *)handle;
    uint32_t rx_hdr_off;
    uint32_t entry_off;

    if (!shim_msgq_is_valid(q) || q->tx_linked || pBackingStore == 0)
    {
        return -1;
    }
    if (msgSize < SHIM_MSGQ_MSG_SIZE_MIN || msgSize > size)
    {
        return -2;
    }
    if (hdrAlign < SHIM_MSGQ_META_MIN_ALIGN ||
        hdrAlign > SHIM_MSGQ_META_MAX_ALIGN ||
        entryAlign < SHIM_MSGQ_META_MIN_ALIGN ||
        entryAlign > SHIM_MSGQ_META_MAX_ALIGN)
    {
        return -3;
    }

    rx_hdr_off = shim_msgq_align_up_u32((uint32_t)sizeof(shim_msgq_tx_header),
                                        hdrAlign);
    entry_off = shim_msgq_align_up_u32(rx_hdr_off +
                                           (uint32_t)sizeof(shim_msgq_rx_header),
                                       entryAlign);
    if (size < entry_off + msgSize)
    {
        return -4;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        return -2000;
    }

    memset(&q->tx, 0, sizeof(q->tx));
    q->tx.version = SHIM_MSGQ_VERSION;
    q->tx.size = size;
    q->tx.msgSize = msgSize;
    q->tx.msgCount = (size - entry_off) / msgSize;
    q->tx.writePtr = 0;
    q->tx.flags = flags;
    q->tx.rxHdrOff = rx_hdr_off;
    q->tx.entryOff = entry_off;

    if (q->tx.msgCount < 2)
    {
        return -5;
    }

    q->our_tx_hdr = (shim_msgq_tx_header *)pBackingStore;
    q->our_rx_hdr = (shim_msgq_rx_header *)((uint8_t *)pBackingStore + rx_hdr_off);
    q->our_entries = (uint8_t *)pBackingStore + entry_off;
    q->tx_linked = 1;
    q->tx_free = q->tx.msgCount - 1U;
    q->rx_swapped = (uint8_t)(((flags & SHIM_MSGQ_FLAGS_SWAP_RX) != 0U) &&
                              ((q->rx.flags & SHIM_MSGQ_FLAGS_SWAP_RX) != 0U));

    memset(q->our_tx_hdr, 0, sizeof(*q->our_tx_hdr));
    memset(q->our_rx_hdr, 0, sizeof(*q->our_rx_hdr));
    memcpy(q->our_tx_hdr, &q->tx, sizeof(q->tx));

    q->tx_create_count++;
    return 0;
}

int msgqRxLink(msgqHandle handle, const void *pBackingStore, unsigned size,
               unsigned msgSize)
{
    shim_msgq *q = (shim_msgq *)handle;
    shim_msgq_tx_header their_tx;

    if (!shim_msgq_is_valid(q) || q->rx_linked || pBackingStore == 0)
    {
        return -1;
    }
    if (msgSize < SHIM_MSGQ_MSG_SIZE_MIN || msgSize > size)
    {
        return -2;
    }

    memcpy(&their_tx, pBackingStore, sizeof(their_tx));
    if (their_tx.version != SHIM_MSGQ_VERSION ||
        their_tx.size != size ||
        their_tx.msgSize != msgSize ||
        their_tx.msgCount == 0 ||
        size < their_tx.entryOff + msgSize)
    {
        return -3;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        return -2000;
    }

    q->rx = their_tx;
    q->their_tx_hdr = (const shim_msgq_tx_header *)pBackingStore;
    q->their_rx_hdr = (shim_msgq_rx_header *)((uint8_t *)pBackingStore +
                                             q->rx.rxHdrOff);
    q->their_entries = (const uint8_t *)pBackingStore + q->rx.entryOff;
    q->rx_read_ptr = 0;
    q->rx_avail = 0;
    q->rx_linked = 1;
    q->rx_swapped = (uint8_t)(((q->tx.flags & SHIM_MSGQ_FLAGS_SWAP_RX) != 0U) &&
                              ((q->rx.flags & SHIM_MSGQ_FLAGS_SWAP_RX) != 0U));

    q->their_rx_hdr->readPtr = 0;
    q->rx_link_count++;
    return 0;
}

unsigned msgqRxGetReadAvailable(msgqHandle handle)
{
    shim_msgq *q = (shim_msgq *)handle;
    uint32_t write_ptr;
    uint32_t avail;

    if (!shim_msgq_is_valid(q) || !q->rx_linked || q->rx.msgCount == 0 ||
        q->their_tx_hdr == 0)
    {
        return 0;
    }

    write_ptr = q->their_tx_hdr->writePtr;
    if (write_ptr >= q->rx.msgCount)
    {
        return 0;
    }

    avail = write_ptr + q->rx.msgCount - q->rx_read_ptr;
    if (avail >= q->rx.msgCount)
    {
        avail -= q->rx.msgCount;
    }

    q->rx.writePtr = write_ptr;
    q->rx_avail = avail;
    return avail;
}

const void *msgqRxGetReadBuffer(msgqHandle handle, unsigned n)
{
    shim_msgq *q = (shim_msgq *)handle;
    uint32_t slot;

    if (!shim_msgq_is_valid(q) || !q->rx_linked || q->their_entries == 0)
    {
        return 0;
    }
    if (n >= q->rx_avail && n >= msgqRxGetReadAvailable(handle))
    {
        return 0;
    }

    slot = shim_msgq_advance(q->rx_read_ptr, q->rx.msgCount, n);
    q->last_read_index = slot;
    q->rx_get_read_count++;
    return q->their_entries + (slot * q->rx.msgSize);
}

int msgqRxMarkConsumed(msgqHandle handle, unsigned n)
{
    shim_msgq *q = (shim_msgq *)handle;

    if (!shim_msgq_is_valid(q) || !q->rx_linked || q->their_rx_hdr == 0)
    {
        return -1;
    }
    if (n > q->rx_avail && n > msgqRxGetReadAvailable(handle))
    {
        return -2;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        return -2000;
    }

    q->rx_read_ptr = shim_msgq_advance(q->rx_read_ptr, q->rx.msgCount, n);
    q->their_rx_hdr->readPtr = q->rx_read_ptr;
    q->rx_avail -= n;
    q->last_rx_consume_count = n;
    q->rx_consume_count++;
    return 0;
}

void *msgqTxGetWriteBuffer(msgqHandle handle, unsigned n)
{
    shim_msgq *q = (shim_msgq *)handle;
    uint32_t slot;

    if (!shim_msgq_is_valid(q) || !q->tx_linked || q->our_entries == 0)
    {
        return 0;
    }
    if (n >= q->tx_free && n >= shim_msgq_tx_free_space(q))
    {
        return 0;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        return 0;
    }

    slot = shim_msgq_advance(q->tx.writePtr, q->tx.msgCount, n);
    q->last_write_index = slot;
    q->tx_get_write_count++;
    return q->our_entries + (slot * q->tx.msgSize);
}

int msgqTxSubmitBuffers(msgqHandle handle, unsigned n)
{
    shim_msgq *q = (shim_msgq *)handle;
    unsigned free_count;

    if (!shim_msgq_is_valid(q) || !q->tx_linked || q->our_tx_hdr == 0)
    {
        return -1;
    }

    free_count = shim_msgq_tx_free_space(q);
    if (n > free_count)
    {
        return -2;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        return -2000;
    }

    q->tx.writePtr = shim_msgq_advance(q->tx.writePtr, q->tx.msgCount, n);
    q->our_tx_hdr->writePtr = q->tx.writePtr;
    q->tx_free = free_count - n;
    q->last_tx_submit_count = n;
    q->tx_submit_count++;
    return 0;
}

int msgqTxSync(msgqHandle handle)
{
    return (int)msgqTxGetFreeSpace(handle);
}

unsigned msgqTxGetPending(msgqHandle handle)
{
    shim_msgq *q = (shim_msgq *)handle;

    if (!shim_msgq_is_valid(q) || !q->tx_linked || q->tx.msgCount == 0)
    {
        return 0;
    }

    return q->tx.msgCount - shim_msgq_tx_free_space(q) - 1U;
}

int msgqRxSync(msgqHandle handle)
{
    return (int)msgqRxGetReadAvailable(handle);
}

int shim_msgqSetOwnershipPlan(msgqHandle handle,
                              const shim_msgq_ownership_plan *plan,
                              uint64_t plan_size)
{
    shim_msgq *q = (shim_msgq *)handle;

    if (!shim_msgq_is_valid(q) || plan == 0 ||
        plan_size < (uint64_t)sizeof(*plan))
    {
        return -1;
    }
    if (plan->version != SHIM_MSGQ_OWNERSHIP_PLAN_VERSION)
    {
        return -2;
    }
    if (!shim_msgq_owner_is_valid(plan->endpoint_owner) ||
        !shim_msgq_owner_is_valid(plan->peer_owner) ||
        !shim_msgq_owner_is_valid(plan->native_queue_owner) ||
        !shim_msgq_owner_is_valid(plan->native_queue_peer))
    {
        return -3;
    }
    if (!shim_msgq_queue_kind_is_valid(plan->native_queue_kind))
    {
        return -4;
    }
    if (!shim_msgq_handoff_state_is_valid(plan->handoff_state))
    {
        return -5;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        return -2000;
    }

    q->endpoint_owner = plan->endpoint_owner;
    q->peer_owner = plan->peer_owner != SHIM_MSGQ_OWNER_UNKNOWN
                        ? plan->peer_owner
                        : shim_msgq_default_peer_owner(plan->endpoint_owner);
    q->native_queue_owner = plan->native_queue_owner;
    q->native_queue_peer = plan->native_queue_peer != SHIM_MSGQ_OWNER_UNKNOWN
                               ? plan->native_queue_peer
                               : shim_msgq_resolved_peer_owner(q);
    q->native_queue_kind = plan->native_queue_kind;
    q->handoff_state = plan->handoff_state;
    q->native_queue_token = plan->native_queue_token;
    return 0;
}

static void shim_msgq_fill_queue_ownership(shim_msgq_queue_ownership *out,
                                           uint32_t producer_owner,
                                           uint32_t consumer_owner,
                                           const shim_msgq_tx_header *hdr,
                                           uint32_t read_ptr,
                                           uint8_t linked)
{
    memset(out, 0, sizeof(*out));
    if (!linked)
    {
        return;
    }

    out->queue_kind = SHIM_MSGQ_QUEUE_KIND_MSGQ;
    out->producer_owner = producer_owner;
    out->consumer_owner = consumer_owner;
    out->msg_count = hdr->msgCount;
    out->msg_size = hdr->msgSize;
    out->write_ptr = hdr->writePtr;
    out->read_ptr = read_ptr;
    out->entry_off = hdr->entryOff;
}

int shim_msgqOwnershipSnapshot(msgqHandle handle,
                               shim_msgq_ownership_snapshot *out,
                               uint64_t out_size)
{
    shim_msgq *q = (shim_msgq *)handle;
    uint32_t peer_owner;

    if (!shim_msgq_is_valid(q) || out == 0 ||
        out_size < (uint64_t)sizeof(*out))
    {
        return -1;
    }

    if (q->rx_linked)
    {
        (void)msgqRxGetReadAvailable(handle);
    }

    peer_owner = shim_msgq_resolved_peer_owner(q);
    memset(out, 0, sizeof(*out));
    out->version = SHIM_MSGQ_OWNERSHIP_SNAPSHOT_VERSION;
    out->endpoint_owner = q->endpoint_owner;
    out->peer_owner = peer_owner;
    out->handoff_state = shim_msgq_resolved_handoff_state(q);
    out->native_queue_owner = q->native_queue_owner;
    out->native_queue_peer = q->native_queue_peer != SHIM_MSGQ_OWNER_UNKNOWN
                                 ? q->native_queue_peer
                                 : peer_owner;
    out->native_queue_kind = q->native_queue_kind;
    out->native_queue_token = q->native_queue_token;

    shim_msgq_fill_queue_ownership(&out->tx, q->endpoint_owner, peer_owner,
                                   &q->tx,
                                   q->our_rx_hdr != 0 ? q->our_rx_hdr->readPtr
                                                      : 0,
                                   q->tx_linked);
    shim_msgq_fill_queue_ownership(&out->rx, peer_owner, q->endpoint_owner,
                                   &q->rx, q->rx_read_ptr, q->rx_linked);
    return 0;
}

int shim_msgqSnapshot(msgqHandle handle, shim_msgq_snapshot *out,
                      uint64_t out_size)
{
    shim_msgq *q = (shim_msgq *)handle;

    if (!shim_msgq_is_valid(q) || out == 0 ||
        out_size < (uint64_t)sizeof(*out))
    {
        return -1;
    }

    if (q->rx_linked)
    {
        (void)msgqRxGetReadAvailable(handle);
    }

    memset(out, 0, sizeof(*out));
    out->version = SHIM_MSGQ_SNAPSHOT_VERSION;
    out->init_count = q->init_count;
    out->tx_linked = q->tx_linked;
    out->rx_linked = q->rx_linked;
    out->rx_swapped = q->rx_swapped;
    out->tx_create_count = q->tx_create_count;
    out->tx_get_write_count = q->tx_get_write_count;
    out->tx_submit_count = q->tx_submit_count;
    out->rx_link_count = q->rx_link_count;
    out->rx_get_read_count = q->rx_get_read_count;
    out->rx_consume_count = q->rx_consume_count;
    out->last_tx_submit_count = q->last_tx_submit_count;
    out->last_rx_consume_count = q->last_rx_consume_count;
    out->tx_write_ptr = q->tx.writePtr;
    out->tx_read_ptr = q->our_rx_hdr != 0 ? q->our_rx_hdr->readPtr : 0;
    out->tx_free = shim_msgq_tx_free_space(q);
    out->tx_msg_count = q->tx.msgCount;
    out->tx_msg_size = q->tx.msgSize;
    out->tx_entry_off = q->tx.entryOff;
    out->rx_write_ptr = q->rx.writePtr;
    out->rx_read_ptr = q->rx_read_ptr;
    out->rx_avail = q->rx_avail;
    out->rx_msg_count = q->rx.msgCount;
    out->rx_msg_size = q->rx.msgSize;
    out->rx_entry_off = q->rx.entryOff;
    return 0;
}

#if !defined(KOT_RM_CLOSURE_LINK) /* superseded by the real RM objects in the closure link */
void *gpumgrGetGpu(uint32_t gpu_id) { (void)gpu_id; return 0; }
int gpuIsCCFeatureEnabled_IMPL(void *gpu) { (void)gpu; return 0; }
int ccslDecryptWithRotationChecks_KERNEL(void) { return -1; }
int ccslEncryptWithRotationChecks_KERNEL(void) { return -1; }
#endif

/*
 * Typed GSP RPC queue shim (solution 101 "Next Work" item 4).
 *
 * A typed layer over the raw msgq above that speaks the canonical GSP
 * message-queue record format, byte-identical to
 * crates/cuda-driverkit/src/gsp_rpc.rs `build_rpc_command_entry` /
 * `drain_rpc_responses` (tinygrad `NVRpcQueue._send_rpc_record`):
 *
 *   [GSP_MSG_QUEUE_ELEMENT 0x30][rpc_message_header_v03 0x20][payload]
 *
 * The element carries checkSum@0x20 (XOR of LE u64 words folded hi32^lo32,
 * computed with checkSum=0), seqNum@0x24, elemCount@0x28. The v03 header
 * carries header_version=3<<24, NV_VGPU_MSG_SIGNATURE_VALID, length =
 * 0x20 + payload, rpc_result/rpc_result_private = RPC_PENDING on commands,
 * sequence = 0 (the element seqNum is the ring sequence).
 *
 * Ownership contract: when the bound msgq's handoff state is
 * NATIVE_RM_PLANNED and this endpoint is not the native RM, command sends
 * are DENIED (-3000) — native RM owns the queue — while poll/consume keep
 * working so the daemon can drain status and decode RPC failures.
 */

#define SHIM_GSP_RPC_MAGIC 0x43505247U /* 'GRPC' little-endian */
#define SHIM_GSP_RPC_SNAPSHOT_VERSION 1U
#define SHIM_GSP_RPC_HEADER_VERSION_V3 0x03000000U
#define SHIM_GSP_RPC_SIGNATURE_VALID 0x43505256U
#define SHIM_GSP_RPC_RESULT_PENDING 0xffffffffU
#define SHIM_GSP_RPC_RESULT_OK 0U
#define SHIM_GSP_RPC_ELEMENT_SIZE 0x30U
#define SHIM_GSP_RPC_HEADER_SIZE 0x20U
#define SHIM_GSP_RPC_RECORD_MIN 0x50U
#define SHIM_GSP_RPC_ELEM_CHECKSUM_OFF 0x20U
#define SHIM_GSP_RPC_ELEM_SEQNUM_OFF 0x24U
#define SHIM_GSP_RPC_ELEM_ELEMCOUNT_OFF 0x28U

#define SHIM_GSP_RPC_KIND_INVALID 0U
#define SHIM_GSP_RPC_KIND_REPLY_PENDING 1U
#define SHIM_GSP_RPC_KIND_REPLY_OK 2U
#define SHIM_GSP_RPC_KIND_REPLY_ERROR 3U
#define SHIM_GSP_RPC_KIND_EVENT 4U

#define SHIM_GSP_RPC_EVENT_FIRST 0x1000U

#define SHIM_GSP_RPC_ERR_OWNER (-3000)

typedef void *gspRpcHandle;

typedef struct
{
    uint32_t kind;
    uint32_t function;
    uint32_t rpc_result;
    uint32_t rpc_result_private;
    uint32_t length;
    uint32_t element_sequence;
    uint32_t element_count;
    uint32_t entries_used;
    uint32_t checksum_ok;
    uint32_t payload_size;
    uint32_t payload_in_entry;
    const void *payload;
} shim_gsp_rpc_record_view;

typedef struct
{
    uint32_t version;
    uint32_t next_sequence;
    uint32_t send_count;
    uint32_t send_denied_admission;
    uint32_t send_denied_owner;
    uint32_t send_reject_count;
    uint32_t poll_count;
    uint32_t poll_empty_count;
    uint32_t consume_count;
    uint32_t invalid_record_count;
    uint32_t checksum_fail_count;
    uint32_t reply_ok_count;
    uint32_t reply_error_count;
    uint32_t reply_pending_count;
    uint32_t event_count;
    uint32_t last_kind;
    uint32_t last_function;
    uint32_t last_result;
    uint32_t last_result_private;
    uint32_t last_sequence;
    uint32_t last_error_function;
    uint32_t last_error_result;
} shim_gsp_rpc_snapshot;

typedef struct
{
    uint32_t magic;
    shim_msgq *queue;
    uint32_t next_sequence;
    uint32_t pending_entries_used;

    uint32_t send_count;
    uint32_t send_denied_admission;
    uint32_t send_denied_owner;
    uint32_t send_reject_count;
    uint32_t poll_count;
    uint32_t poll_empty_count;
    uint32_t consume_count;
    uint32_t invalid_record_count;
    uint32_t checksum_fail_count;
    uint32_t reply_ok_count;
    uint32_t reply_error_count;
    uint32_t reply_pending_count;
    uint32_t event_count;
    uint32_t last_kind;
    uint32_t last_function;
    uint32_t last_result;
    uint32_t last_result_private;
    uint32_t last_sequence;
    uint32_t last_error_function;
    uint32_t last_error_result;
} shim_gsp_rpc_ctx;

static uint32_t shim_gsp_rpc_read_u32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static void shim_gsp_rpc_write_u32(uint8_t *bytes, uint32_t value)
{
    bytes[0] = (uint8_t)(value & 0xffU);
    bytes[1] = (uint8_t)((value >> 8) & 0xffU);
    bytes[2] = (uint8_t)((value >> 16) & 0xffU);
    bytes[3] = (uint8_t)((value >> 24) & 0xffU);
}

/* tinygrad NVRpcQueue._checksum: zero-pad to 8, XOR LE u64 words, fold hi^lo.
 * The element checkSum field must be zero while this runs. */
static uint32_t shim_gsp_rpc_checksum(const uint8_t *data, uint32_t len)
{
    uint64_t acc = 0;
    uint32_t i = 0;

    while (i < len)
    {
        uint64_t word = 0;
        uint32_t n = len - i < 8U ? len - i : 8U;
        for (uint32_t b = 0; b < n; b++)
        {
            word |= (uint64_t)data[i + b] << (8U * b);
        }
        acc ^= word;
        i += 8U;
    }
    return (uint32_t)(acc >> 32) ^ (uint32_t)acc;
}

static shim_gsp_rpc_ctx *shim_gsp_rpc_ctx_from_handle(gspRpcHandle handle)
{
    shim_gsp_rpc_ctx *ctx = (shim_gsp_rpc_ctx *)handle;

    if (ctx == 0 || ctx->magic != SHIM_GSP_RPC_MAGIC ||
        !shim_msgq_is_valid(ctx->queue))
    {
        return 0;
    }
    return ctx;
}

/* Native RM owns the queue: only the native RM endpoint may produce commands;
 * everyone may still drain. */
static int shim_gsp_rpc_send_owner_denied(const shim_msgq *q)
{
    return shim_msgq_resolved_handoff_state(q) ==
               SHIM_MSGQ_HANDOFF_NATIVE_RM_PLANNED &&
           q->endpoint_owner != SHIM_MSGQ_OWNER_NATIVE_RM;
}

unsigned shim_gspRpcCtxSize(void)
{
    return (unsigned)sizeof(shim_gsp_rpc_ctx);
}

int shim_gspRpcInit(gspRpcHandle *pHandle, void *pBuffer, msgqHandle queue)
{
    shim_gsp_rpc_ctx *ctx = (shim_gsp_rpc_ctx *)pBuffer;

    if (ctx == 0 || !shim_msgq_is_valid((shim_msgq *)queue))
    {
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->magic = SHIM_GSP_RPC_MAGIC;
    ctx->queue = (shim_msgq *)queue;

    if (pHandle != 0)
    {
        *pHandle = ctx;
    }
    return 0;
}

static int shim_gsp_rpc_send_record(shim_gsp_rpc_ctx *ctx, uint32_t function,
                                    uint32_t rpc_result,
                                    uint32_t rpc_result_private,
                                    const void *payload, uint32_t payload_size)
{
    shim_msgq *q = ctx->queue;
    uint8_t *record;
    uint32_t record_len;
    uint32_t csum;

    if (!q->tx_linked)
    {
        return -2;
    }
    if (q->tx.msgSize < SHIM_GSP_RPC_RECORD_MIN ||
        payload_size > q->tx.msgSize - SHIM_GSP_RPC_RECORD_MIN)
    {
        ctx->send_reject_count++;
        return -3;
    }
    if (payload_size != 0 && payload == 0)
    {
        return -1;
    }
    if (shim_gsp_rpc_send_owner_denied(q))
    {
        ctx->send_denied_owner++;
        return SHIM_GSP_RPC_ERR_OWNER;
    }
    if (shim_mutation_admit(CUDK_MUTATION_OP_GSP_QUEUE_SUBMIT) != 0)
    {
        ctx->send_denied_admission++;
        return -2000;
    }

    record = msgqTxGetWriteBuffer(q, 0);
    if (record == 0)
    {
        return -4;
    }

    record_len = SHIM_GSP_RPC_RECORD_MIN + payload_size;
    memset(record, 0, q->tx.msgSize);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEM_SEQNUM_OFF,
                           ctx->next_sequence);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEM_ELEMCOUNT_OFF, 1U);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE,
                           SHIM_GSP_RPC_HEADER_VERSION_V3);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x04,
                           SHIM_GSP_RPC_SIGNATURE_VALID);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x08,
                           SHIM_GSP_RPC_HEADER_SIZE + payload_size);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x0c, function);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x10,
                           rpc_result);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x14,
                           rpc_result_private);
    /* v03 header sequence stays 0; the element seqNum is the ring sequence. */
    if (payload_size != 0)
    {
        memcpy(record + SHIM_GSP_RPC_RECORD_MIN, payload, payload_size);
    }

    csum = shim_gsp_rpc_checksum(record, record_len);
    shim_gsp_rpc_write_u32(record + SHIM_GSP_RPC_ELEM_CHECKSUM_OFF, csum);

    if (msgqTxSubmitBuffers(q, 1) != 0)
    {
        return -5;
    }

    ctx->next_sequence++;
    ctx->send_count++;
    return 0;
}

int shim_gspRpcSend(gspRpcHandle handle, uint32_t function,
                    const void *payload, uint32_t payload_size)
{
    shim_gsp_rpc_ctx *ctx = shim_gsp_rpc_ctx_from_handle(handle);

    if (ctx == 0)
    {
        return -1;
    }
    return shim_gsp_rpc_send_record(ctx, function,
                                    SHIM_GSP_RPC_RESULT_PENDING,
                                    SHIM_GSP_RPC_RESULT_PENDING, payload,
                                    payload_size);
}

/* GSP→host status-queue producer face (reply/event records carry results). */
int shim_gspRpcSendStatus(gspRpcHandle handle, uint32_t function,
                          uint32_t rpc_result, uint32_t rpc_result_private,
                          const void *payload, uint32_t payload_size)
{
    shim_gsp_rpc_ctx *ctx = shim_gsp_rpc_ctx_from_handle(handle);

    if (ctx == 0)
    {
        return -1;
    }
    return shim_gsp_rpc_send_record(ctx, function, rpc_result,
                                    rpc_result_private, payload, payload_size);
}

int shim_gspRpcPoll(gspRpcHandle handle, shim_gsp_rpc_record_view *out)
{
    shim_gsp_rpc_ctx *ctx = shim_gsp_rpc_ctx_from_handle(handle);
    shim_msgq *q;
    const uint8_t *record;
    uint32_t msg_size;
    uint32_t avail;
    uint32_t stored_csum;
    uint32_t record_len;
    uint32_t total_len;

    if (ctx == 0 || out == 0)
    {
        return -1;
    }
    q = ctx->queue;
    if (!q->rx_linked)
    {
        return -2;
    }

    memset(out, 0, sizeof(*out));
    ctx->poll_count++;

    avail = msgqRxGetReadAvailable(q);
    if (avail == 0)
    {
        ctx->poll_empty_count++;
        ctx->pending_entries_used = 0;
        return 1;
    }

    record = msgqRxGetReadBuffer(q, 0);
    if (record == 0)
    {
        ctx->poll_empty_count++;
        ctx->pending_entries_used = 0;
        return 1;
    }

    msg_size = q->rx.msgSize;
    out->element_sequence =
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEM_SEQNUM_OFF);
    out->element_count =
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEM_ELEMCOUNT_OFF);
    out->length = shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x08);
    out->function =
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x0c);
    out->rpc_result =
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x10);
    out->rpc_result_private =
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x14);

    total_len = SHIM_GSP_RPC_ELEMENT_SIZE + out->length;
    out->entries_used =
        msg_size != 0 ? (total_len + msg_size - 1U) / msg_size : 1U;
    if (out->entries_used == 0)
    {
        out->entries_used = 1;
    }

    if (msg_size < SHIM_GSP_RPC_RECORD_MIN ||
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE + 0x04) !=
            SHIM_GSP_RPC_SIGNATURE_VALID ||
        shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEMENT_SIZE) !=
            SHIM_GSP_RPC_HEADER_VERSION_V3 ||
        out->length < SHIM_GSP_RPC_HEADER_SIZE || out->entries_used > avail)
    {
        out->kind = SHIM_GSP_RPC_KIND_INVALID;
        out->entries_used = 1;
        ctx->invalid_record_count++;
        ctx->pending_entries_used = 1;
        ctx->last_kind = out->kind;
        return 0;
    }

    out->payload_size = out->length - SHIM_GSP_RPC_HEADER_SIZE;
    out->payload = record + SHIM_GSP_RPC_RECORD_MIN;
    out->payload_in_entry = out->payload_size;
    if (out->payload_in_entry > msg_size - SHIM_GSP_RPC_RECORD_MIN)
    {
        /* Continuation entries are ring-resident, not contiguous here. */
        out->payload_in_entry = msg_size - SHIM_GSP_RPC_RECORD_MIN;
    }

    /* Verify the tinygrad checksum without copying: the stored checkSum sits
     * in the low 4 bytes of the LE u64 word at 0x20, so XOR-ing it out of the
     * folded accumulator recovers the checkSum=0 form. Single-entry records
     * only; spanning records are left unchecked (checksum_ok = 2). */
    record_len = total_len;
    if (record_len <= msg_size)
    {
        uint32_t computed = shim_gsp_rpc_checksum(record, record_len);
        stored_csum =
            shim_gsp_rpc_read_u32(record + SHIM_GSP_RPC_ELEM_CHECKSUM_OFF);
        computed ^= stored_csum;
        out->checksum_ok = computed == stored_csum ? 1U : 0U;
        if (out->checksum_ok == 0)
        {
            ctx->checksum_fail_count++;
        }
    }
    else
    {
        out->checksum_ok = 2;
    }

    if (out->function >= SHIM_GSP_RPC_EVENT_FIRST)
    {
        out->kind = SHIM_GSP_RPC_KIND_EVENT;
        ctx->event_count++;
    }
    else if (out->rpc_result == SHIM_GSP_RPC_RESULT_PENDING)
    {
        out->kind = SHIM_GSP_RPC_KIND_REPLY_PENDING;
        ctx->reply_pending_count++;
    }
    else if (out->rpc_result == SHIM_GSP_RPC_RESULT_OK)
    {
        out->kind = SHIM_GSP_RPC_KIND_REPLY_OK;
        ctx->reply_ok_count++;
    }
    else
    {
        out->kind = SHIM_GSP_RPC_KIND_REPLY_ERROR;
        ctx->reply_error_count++;
        ctx->last_error_function = out->function;
        ctx->last_error_result = out->rpc_result;
    }

    ctx->pending_entries_used = out->entries_used;
    ctx->last_kind = out->kind;
    ctx->last_function = out->function;
    ctx->last_result = out->rpc_result;
    ctx->last_result_private = out->rpc_result_private;
    ctx->last_sequence = out->element_sequence;
    return 0;
}

int shim_gspRpcConsume(gspRpcHandle handle)
{
    shim_gsp_rpc_ctx *ctx = shim_gsp_rpc_ctx_from_handle(handle);
    int rc;

    if (ctx == 0)
    {
        return -1;
    }
    if (ctx->pending_entries_used == 0)
    {
        return -2;
    }

    rc = msgqRxMarkConsumed(ctx->queue, ctx->pending_entries_used);
    if (rc != 0)
    {
        return rc;
    }
    ctx->pending_entries_used = 0;
    ctx->consume_count++;
    return 0;
}

int shim_gspRpcSnapshot(gspRpcHandle handle, shim_gsp_rpc_snapshot *out,
                        uint64_t out_size)
{
    shim_gsp_rpc_ctx *ctx = shim_gsp_rpc_ctx_from_handle(handle);

    if (ctx == 0 || out == 0 || out_size < (uint64_t)sizeof(*out))
    {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->version = SHIM_GSP_RPC_SNAPSHOT_VERSION;
    out->next_sequence = ctx->next_sequence;
    out->send_count = ctx->send_count;
    out->send_denied_admission = ctx->send_denied_admission;
    out->send_denied_owner = ctx->send_denied_owner;
    out->send_reject_count = ctx->send_reject_count;
    out->poll_count = ctx->poll_count;
    out->poll_empty_count = ctx->poll_empty_count;
    out->consume_count = ctx->consume_count;
    out->invalid_record_count = ctx->invalid_record_count;
    out->checksum_fail_count = ctx->checksum_fail_count;
    out->reply_ok_count = ctx->reply_ok_count;
    out->reply_error_count = ctx->reply_error_count;
    out->reply_pending_count = ctx->reply_pending_count;
    out->event_count = ctx->event_count;
    out->last_kind = ctx->last_kind;
    out->last_function = ctx->last_function;
    out->last_result = ctx->last_result;
    out->last_result_private = ctx->last_result_private;
    out->last_sequence = ctx->last_sequence;
    out->last_error_function = ctx->last_error_function;
    out->last_error_result = ctx->last_error_result;
    return 0;
}

const char *shim_gspRpcFunctionName(uint32_t function)
{
    switch (function)
    {
    case 4: return "ALLOC_MEMORY";
    case 54: return "SET_PAGE_DIRECTORY";
    case 72: return "GSP_SET_SYSTEM_INFO";
    case 73: return "SET_REGISTRY";
    case 76: return "GSP_RM_CONTROL";
    case 103: return "GSP_RM_ALLOC";
    case 0x1001: return "GSP_INIT_DONE";
    case 0x1002: return "GSP_RUN_CPU_SEQUENCER";
    case 0x1003: return "RC_TRIGGERED";
    case 0x1004: return "MMU_FAULT";
    case 0x100c: return "UCODE_LIBOS_PRINT";
    case 0x101c: return "GSP_LOCKDOWN_NOTICE";
    case 0x1020: return "GSP_POST_NOCAT_RECORD";
    default: return "UNKNOWN";
    }
}
#endif /* !KOT_RM_CLOSURE_LINK (msgq + typed GSP RPC shim) */
