/*
 * SPDX-License-Identifier: MIT
 *
 * cuda_runtime_api.h — macOS arm64e CUDA *Runtime API* shim header.
 *
 * Declares the runtime-API TYPES and FUNCTION prototypes that MLX's CUDA
 * backend (mlx/backend/cuda) references, so that backend can be compiled and
 * linked against libcudart.dylib on macOS. The runtime calls are thunked onto
 * cuda-driverkit's CUDA *Driver* API (libcuda_driverkit.dylib) — see
 * portkit/cudart-shim/src/lib.rs for the live-thunked vs. honestly-stubbed
 * split.
 *
 * Layout note: MLX aggregate-initializes cudaLaunchConfig_t / cudaLaunchAttribute
 * / cudaKernelNodeParams / cudaDeviceProp by named field and reads specific
 * members, so those structs mirror the upstream CUDA 12.x field order. The
 * cudaDevAttr* / cudaMemcpyKind / cudaError enum values match upstream cudart.
 */

#ifndef CUDART_SHIM_CUDA_RUNTIME_API_H
#define CUDART_SHIM_CUDA_RUNTIME_API_H

/* Report a CUDA 12.x runtime so MLX's `#if CUDART_VERSION >= 12000` branches take
 * the modern path that matches this shim's API signatures (otherwise an undefined
 * CUDART_VERSION = 0 selects the legacy pre-12 graph signatures → "no matching
 * function"). MLX's CUDA backend targets 12.x. */
#ifndef CUDART_VERSION
#define CUDART_VERSION 12080
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  Error codes (cudaError / cudaError_t)
 *  Values match upstream cudart so MLX's `== cudaSuccess` comparisons and
 *  error-code propagation behave identically.
 * ===================================================================== */
typedef enum cudaError {
    cudaSuccess                        = 0,
    cudaErrorInvalidValue              = 1,
    cudaErrorMemoryAllocation          = 2,
    cudaErrorInitializationError       = 3,
    cudaErrorInsufficientDriver        = 35,
    cudaErrorNoDevice                  = 100,
    cudaErrorInvalidDevice             = 101,
    cudaErrorPeerAccessAlreadyEnabled  = 704,
    cudaErrorInvalidResourceHandle     = 400,
    cudaErrorNotReady                  = 600,
    cudaErrorNotSupported              = 801,
    cudaErrorUnknown                   = 999,
} cudaError_t;

/* ===================================================================== *
 *  Opaque handle types
 * ===================================================================== */
typedef struct CUstream_st*  cudaStream_t;
typedef struct CUevent_st*   cudaEvent_t;
typedef struct CUgraph_st*   cudaGraph_t;
typedef struct CUgraphNode_st* cudaGraphNode_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;
typedef struct CUmemPoolHandle_st* cudaMemPool_t;

/* cudaUUID_t — 16 raw bytes, as upstream. */
typedef struct cudaUUID_t {
    char bytes[16];
} cudaUUID_t;

/* cudaDataType_t — single definition lives in library_types.h (shared with the
 * cuBLAS/cuFFT/cuBLASLt stubs). Including it here (instead of re-declaring the
 * enum) avoids an ODR clash when a TU pulls both <cuda_runtime.h> and
 * <cublasLt.h>, and gives the cudart surface the full dtype set (FP8/FP6/FP4).
 * Resolved from the unified toolkit-root include dir (or any -I/-isystem that
 * carries the mathlib-stubs headers). */
#include "library_types.h"

/* ===================================================================== *
 *  Enums referenced by MLX (values match upstream cudart)
 * ===================================================================== */
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} cudaMemcpyKind;

typedef enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost         = 1,
    cudaMemoryTypeDevice       = 2,
    cudaMemoryTypeManaged      = 3,
} cudaMemoryType;

typedef struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int                 device;
    void*               devicePointer;
    void*               hostPointer;
} cudaPointerAttributes;

/* cudaDeviceAttr — only the ids MLX queries are enumerated (upstream values). */
typedef enum cudaDeviceAttr {
    cudaDevAttrComputeCapabilityMajor    = 75,
    cudaDevAttrComputeCapabilityMinor    = 76,
    cudaDevAttrConcurrentManagedAccess   = 89,
    cudaDevAttrManagedMemory             = 83,
    cudaDevAttrMemoryPoolsSupported      = 115,
    cudaDevAttrHostNativeAtomicSupported = 86,
} cudaDeviceAttr;

/* Event creation / record flags (upstream values). */
#define cudaEventDefault       0x00
#define cudaEventBlockingSync  0x01
#define cudaEventDisableTiming 0x02

/* Stream creation flags (upstream values). */
#define cudaStreamDefault      0x00
#define cudaStreamNonBlocking  0x01

typedef enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
} cudaFuncAttribute;

typedef enum cudaMemoryAdvise {
    cudaMemAdviseSetReadMostly  = 1,
    cudaMemAdviseSetAccessedBy  = 5,
} cudaMemoryAdvise;

typedef enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice  = 1,
} cudaMemLocationType;

typedef struct cudaMemLocation {
    cudaMemLocationType type;
    int                 id;
} cudaMemLocation;

typedef enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies   = 0x1,
    cudaMemPoolReuseAllowOpportunistic        = 0x2,
    cudaMemPoolReuseAllowInternalDependencies = 0x3,
    cudaMemPoolAttrReleaseThreshold           = 0x4,
    cudaMemPoolAttrReservedMemCurrent         = 0x5,
    cudaMemPoolAttrReservedMemHigh            = 0x6,
    cudaMemPoolAttrUsedMemCurrent             = 0x7,
    cudaMemPoolAttrUsedMemHigh                = 0x8,
} cudaMemPoolAttr;

/* cudaMemAccessDesc / flags (cudaMemPoolSetAccess) */
typedef enum cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone      = 0,
    cudaMemAccessFlagsProtRead      = 1,
    cudaMemAccessFlagsProtReadWrite = 3,
} cudaMemAccessFlags;
typedef struct cudaMemAccessDesc {
    cudaMemLocation    location;
    cudaMemAccessFlags flags;
} cudaMemAccessDesc;

/* ===================================================================== *
 *  IPC handle types (cudaIpc* — c10/cuda CUDACachingAllocator + CUDAEvent).
 *  Sizes match upstream cudart (64-byte opaque blobs). Stubbed at the API.
 * ===================================================================== */
#define CUDA_IPC_HANDLE_SIZE 64
typedef struct cudaIpcEventHandle_st {
    char reserved[CUDA_IPC_HANDLE_SIZE];
} cudaIpcEventHandle_t;
typedef struct cudaIpcMemHandle_st {
    char reserved[CUDA_IPC_HANDLE_SIZE];
} cudaIpcMemHandle_t;
#define cudaIpcMemLazyEnablePeerAccess 0x01

/* Event-record flags (cudaEventRecordWithFlags). */
#define cudaEventRecordDefault  0x00
#define cudaEventRecordExternal 0x01

/* Event-wait flags (cudaStreamWaitEvent via CUDAEvent::block). */
#define cudaEventWaitDefault  0x00
#define cudaEventWaitExternal 0x01

/* Extra event-create flag referenced by c10's CUDAEvent. */
#define cudaEventInterprocess 0x04

/* Stream-capture status / mode (graph capture; stubbed at the API). */
typedef enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone        = 0,
    cudaStreamCaptureStatusActive      = 1,
    cudaStreamCaptureStatusInvalidated = 2,
} cudaStreamCaptureStatus;

typedef enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2,
} cudaStreamCaptureMode;

typedef enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel      = 0,
    cudaGraphNodeTypeMemcpy      = 1,
    cudaGraphNodeTypeMemset      = 2,
    cudaGraphNodeTypeHost        = 3,
    cudaGraphNodeTypeGraph       = 4,
    cudaGraphNodeTypeEmpty       = 5,
    cudaGraphNodeTypeWaitEvent   = 6,
    cudaGraphNodeTypeEventRecord = 7,
} cudaGraphNodeType;

typedef enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError   = 1,
} cudaGraphExecUpdateResult;

typedef struct cudaGraphExecUpdateResultInfo {
    cudaGraphExecUpdateResult result;
    cudaGraphNode_t           errorNode;
    cudaGraphNode_t           errorFromNode;
} cudaGraphExecUpdateResultInfo;

/* cudaMemsetParams — referenced in cudnn-frontend's graph-node signatures. */
typedef struct cudaMemsetParams {
    void*        dst;
    size_t       pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t       width;
    size_t       height;
} cudaMemsetParams;

/* Graph edge data (cudaGraphNodeGetDependencies overload). */
typedef struct cudaGraphEdgeData {
    unsigned char from_port;
    unsigned char to_port;
    unsigned char type;
    unsigned char reserved[5];
} cudaGraphEdgeData;

/* Driver-entry-point query (cudaGetDriverEntryPoint*). */
typedef enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess           = 0,
    cudaDriverEntryPointSymbolNotFound    = 1,
    cudaDriverEntryPointVersionNotSufficient = 2,
} cudaDriverEntryPointQueryResult;
#define cudaEnableDefault 0x0

/* Kernel-node attribute id/value used by graph attribute set/get (stubbed). */
typedef enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore           = 0,
    cudaLaunchAttributeClusterDimension = 4,
} cudaLaunchAttributeID;

/* ===================================================================== *
 *  dim3 — matches upstream; MLX builds these by value.
 * ===================================================================== */
#ifndef CUDART_SHIM_HAS_DIM3
#define CUDART_SHIM_HAS_DIM3
struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
#endif
};
typedef struct dim3 dim3;
#endif

/* ===================================================================== *
 *  Launch config / attributes (cudaLaunchKernelExC path)
 *  Field order/names mirror upstream so MLX's aggregate init compiles and the
 *  members it sets (.gridDim/.blockDim/.dynamicSmemBytes/.stream/.attrs/
 *  .numAttrs and attr.id / attr.val.clusterDim.*) resolve correctly.
 * ===================================================================== */
typedef struct cudaLaunchAttributeValue {
    union {
        struct { unsigned int x, y, z; } clusterDim;
        unsigned char pad[64];
    };
} cudaLaunchAttributeValue;

typedef struct cudaLaunchAttribute {
    cudaLaunchAttributeID    id;
    cudaLaunchAttributeValue val;
} cudaLaunchAttribute;

/* Kernel-node attribute value alias (used by cudaGraphKernelNodeSet/GetAttribute) */
typedef cudaLaunchAttributeValue cudaKernelNodeAttrValue;

typedef struct cudaLaunchConfig_st {
    dim3                 gridDim;
    dim3                 blockDim;
    size_t               dynamicSmemBytes;
    cudaStream_t         stream;
    cudaLaunchAttribute* attrs;
    unsigned int         numAttrs;
} cudaLaunchConfig_t;

/* cudaKernelNodeParams — populated by MLX's graph path (.func/.gridDim/...). */
typedef struct cudaKernelNodeParams {
    void*        func;
    dim3         gridDim;
    dim3         blockDim;
    unsigned int sharedMemBytes;
    void**       kernelParams;
    void**       extra;
} cudaKernelNodeParams;

/* ===================================================================== *
 *  cudaDeviceProp — only the members MLX reads are laid out faithfully; the
 *  tail is padded so sizeof() comfortably exceeds what callers touch. MLX
 *  reads: name, uuid, major, minor, pciDomainID, pciBusID, pciDeviceID.
 * ===================================================================== */
typedef struct cudaDeviceProp {
    char       name[256];
    cudaUUID_t uuid;
    int        major;
    int        minor;
    int        pciDomainID;
    int        pciBusID;
    int        pciDeviceID;
    int        multiProcessorCount;
    size_t     totalGlobalMem;
    /* Padding to stay >= upstream sizeof so any stray field access is in-bounds. */
    unsigned char _reserved[1024];
} cudaDeviceProp;

/* host callback signature (cudaLaunchHostFunc) */
typedef void (*cudaHostFn_t)(void* userData);

/* ===================================================================== *
 *  FUNCTION prototypes — runtime API surface MLX calls.
 *  See src/lib.rs for the live-thunked vs. stubbed disposition of each.
 * ===================================================================== */

/* --- error reporting --- */
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);

/* --- device management --- */
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

/* --- IPC (stubbed -> cudaErrorNotSupported) --- */
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr);
cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
cudaError_t cudaIpcCloseMemHandle(void* devPtr);
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event);
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle);

/* --- memory --- */
cudaError_t cudaMalloc(void** devPtr, size_t size);
#ifndef cudaMemAttachGlobal
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost   0x02
#define cudaMemAttachSingle 0x04
#endif
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags
#ifdef __cplusplus
    = cudaMemAttachGlobal
#endif
    );
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr);
cudaError_t cudaMemAdvise(const void* devPtr, size_t count,
                          enum cudaMemoryAdvise advice, int device);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value);
cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count);
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                                size_t count, cudaStream_t stream);

/* --- stream --- */
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus);
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus,
                                     unsigned long long* pId);
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus,
                                        unsigned long long* pId, cudaGraph_t* pGraph,
                                        const cudaGraphNode_t** pDependencies, size_t* pNumDependencies);
cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode);

/* --- driver entry point lookup (stubbed) --- */
cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                    enum cudaDriverEntryPointQueryResult* driverStatus);
cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion,
                                             unsigned long long flags,
                                             enum cudaDriverEntryPointQueryResult* driverStatus);

/* --- event --- */
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

/* --- launch / function attributes / occupancy --- */
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args);
cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value);
cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, const void* func);

/* --- CUDA graph API (ALL stubbed -> cudaErrorNotSupported; see src/lib.rs) --- */
cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
/* MLX (utils.cpp) calls the legacy 5-arg form unconditionally; provide it (the
 * pErrorNode/pLogBuffer/bufferSize args are accepted + ignored by the stub). */
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                 cudaGraphNode_t* pErrorNode, char* pLogBuffer,
                                 size_t bufferSize);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t graphExec, cudaGraph_t graph,
                                cudaGraphExecUpdateResultInfo* resultInfo);
cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t* deps, size_t numDeps,
                                   const cudaKernelNodeParams* nodeParams);
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                  const cudaGraphNode_t* deps, size_t numDeps);
cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                       const cudaGraphNode_t* deps, size_t numDeps,
                                       cudaGraph_t childGraph);
cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                     const cudaGraphNode_t* to, size_t numDeps);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* pType);
cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph);
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies,
                                         cudaGraphEdgeData* edgeData, size_t* pNumDependencies);
cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t node, enum cudaLaunchAttributeID attr,
                                            const cudaKernelNodeAttrValue* value);
cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t node, enum cudaLaunchAttributeID attr,
                                            cudaKernelNodeAttrValue* value);
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags);

/* --- runtime graph/stream/memset fns cudnn-frontend's shim calls (stubbed) --- */
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes);
cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes,
                                           size_t* pNumDependentNodes);
cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                     const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                     void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src,
                                           size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                   const cudaMemsetParams* pMemsetParams);
cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams);

#ifdef __cplusplus
}
#endif

#endif /* CUDART_SHIM_CUDA_RUNTIME_API_H */
