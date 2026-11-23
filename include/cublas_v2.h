// cublas_v2.h — minimal macOS arm64e LINK-STUB header for MLX's CUDA backend.
//
// Declares ONLY the cuBLAS surface MLX's mlx/backend/cuda actually references.
// Pairs with libcublas.dylib (honest no-op: every fn returns
// CUBLAS_STATUS_NOT_SUPPORTED). Real BLAS path is MathDx/cuBLASDx
// (portkit solution 98) or sp3 Linux nvidia.ko — out of scope here.
#ifndef MLX_STUB_CUBLAS_V2_H
#define MLX_STUB_CUBLAS_V2_H

#include <stddef.h>
#include "library_types.h" // cudaDataType_t

#ifdef __cplusplus
extern "C" {
#endif

// --- status enum (only the values MLX compares against) ---
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_NOT_SUPPORTED = 15
} cublasStatus_t;

// --- compute type + op enums MLX passes through ---
typedef enum {
  CUBLAS_COMPUTE_16F = 64,
  CUBLAS_COMPUTE_32F = 68,
  CUBLAS_COMPUTE_32F_FAST_16F = 74,
  CUBLAS_COMPUTE_32F_FAST_16BF = 75,
  CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
  CUBLAS_COMPUTE_64F = 70
} cublasComputeType_t;

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;

// MLX references cublasGetStatusString (and cublasLt* via cublasLt.h).
const char* cublasGetStatusString(cublasStatus_t status);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLX_STUB_CUBLAS_V2_H
