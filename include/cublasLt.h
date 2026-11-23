// cublasLt.h — minimal macOS arm64e LINK-STUB header for MLX's CUDA backend.
//
// Declares ONLY the cuBLASLt surface MLX (matmul.cpp / cublas_utils.cpp)
// references. Pairs with libcublasLt.dylib (honest no-op: every fn returns
// CUBLAS_STATUS_NOT_SUPPORTED). Real BLAS path is MathDx/cuBLASDx
// (portkit solution 98) — out of scope here.
#ifndef MLX_STUB_CUBLASLT_H
#define MLX_STUB_CUBLASLT_H

#include <stddef.h>
#include <stdint.h>
#include "cublas_v2.h"       // cublasStatus_t, cublasComputeType_t, cublasOperation_t
#include "library_types.h"   // cudaDataType_t

#ifdef __cplusplus
extern "C" {
#endif

// --- opaque handle/descriptor types (pointers) ---
typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct cublasLtMatmulDescOpaque* cublasLtMatmulDesc_t;
typedef struct cublasLtMatrixLayoutOpaque* cublasLtMatrixLayout_t;
typedef struct cublasLtMatmulPreferenceOpaque* cublasLtMatmulPreference_t;

// MLX names these types but treats their values opaquely (the stub path is a
// no-op returning NOT_SUPPORTED, so only distinct names — not exact values —
// matter; real values are kept where known).
typedef enum {
  CUBLASLT_POINTER_MODE_HOST = 0,
  CUBLASLT_POINTER_MODE_DEVICE = 1
} cublasLtPointerMode_t;
typedef enum {
  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F = 0,
  CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 = 1,  // Blackwell MX block scaling
  CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 = 2
} cublasLtMatmulMatrixScale_t;
// Matrix memory order + batch mode (MLX sets/compares these opaquely).
typedef enum { CUBLASLT_ORDER_COL = 0, CUBLASLT_ORDER_ROW = 1 } cublasLtOrder_t;
typedef enum {
  CUBLASLT_BATCH_MODE_STRIDED = 0,
  CUBLASLT_BATCH_MODE_POINTER_ARRAY = 1
} cublasLtBatchMode_t;

// Epilogue: MLX uses CUBLASLT_EPILOGUE_BIAS.
typedef enum { CUBLASLT_EPILOGUE_DEFAULT = 1, CUBLASLT_EPILOGUE_BIAS = 4 } cublasLtEpilogue_t;

// --- attribute enums (only the members MLX sets) ---
typedef enum {
  CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
  CUBLASLT_MATMUL_DESC_TRANSA = 3,
  CUBLASLT_MATMUL_DESC_TRANSB = 4,
  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
  CUBLASLT_MATMUL_DESC_EPILOGUE = 23,
  CUBLASLT_MATMUL_DESC_BIAS_POINTER = 24,
  CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 100,  // newest (Blackwell); opaque to stub
  CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 101
} cublasLtMatmulDescAttributes_t;

typedef enum {
  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 4,
  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 5,
  CUBLASLT_MATRIX_LAYOUT_BATCH_MODE = 6
} cublasLtMatrixLayoutAttribute_t;

typedef enum {
  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
} cublasLtMatmulPreferenceAttributes_t;

// Opaque algo + the heuristic result struct MLX reads (.state/.algo/.workspaceSize).
typedef struct {
  uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
  cublasLtMatmulAlgo_t algo;
  size_t workspaceSize;
  cublasStatus_t state;
  float wavesCount;
  int reserved[4];
} cublasLtMatmulHeuristicResult_t;

// Forward-decl for cudaStream_t (opaque) without dragging cuda_runtime.h.
typedef struct CUstream_st* cudaStream_t;

// --- functions MLX calls. cublasLtMatmul (fp32, non-batched, no-bias) computes
//     for REAL via Accelerate cblas_sgemm; other configs return NOT_SUPPORTED. ---
cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);
size_t cublasLtGetVersion(void);
size_t cublasLtGetCudartVersion(void);
const char* cublasLtGetStatusString(cublasStatus_t status);

cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t* matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);

cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t* matLayout,
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* buf,
    size_t sizeInBytes);

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t* heuristicResultsArray,
    int* returnAlgoCount);

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void* alpha,
    const void* A,
    cublasLtMatrixLayout_t Adesc,
    const void* B,
    cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C,
    cublasLtMatrixLayout_t Cdesc,
    void* D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLX_STUB_CUBLASLT_H
