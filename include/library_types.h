// library_types.h — macOS arm64e LINK-STUB header for MLX's CUDA backend.
// Single source of truth for cudaDataType_t (the cudart shim's
// cuda_runtime_api.h #includes this rather than re-declaring it, so a TU that
// pulls both <cuda_runtime.h> and <cublasLt.h> has no ODR clash). The full
// enumerator set + values are transcribed verbatim from NVIDIA's real
// library_types.h (CUDA 12.8 redist) so every dtype MLX names — including the
// Blackwell FP8/FP6/FP4 microscaling types — resolves with the correct value.
#ifndef __LIBRARY_TYPES_H__
#define __LIBRARY_TYPES_H__

typedef enum cudaDataType_t {
  CUDA_R_16F      =  2,  // real fp16
  CUDA_C_16F      =  6,  // complex fp16
  CUDA_R_16BF     = 14,  // real bf16
  CUDA_C_16BF     = 15,  // complex bf16
  CUDA_R_32F      =  0,  // real fp32
  CUDA_C_32F      =  4,  // complex fp32
  CUDA_R_64F      =  1,  // real fp64
  CUDA_C_64F      =  5,  // complex fp64
  CUDA_R_4I       = 16,
  CUDA_C_4I       = 17,
  CUDA_R_4U       = 18,
  CUDA_C_4U       = 19,
  CUDA_R_8I       =  3,
  CUDA_C_8I       =  7,
  CUDA_R_8U       =  8,
  CUDA_C_8U       =  9,
  CUDA_R_16I      = 20,
  CUDA_C_16I      = 21,
  CUDA_R_16U      = 22,
  CUDA_C_16U      = 23,
  CUDA_R_32I      = 10,
  CUDA_C_32I      = 11,
  CUDA_R_32U      = 12,
  CUDA_C_32U      = 13,
  CUDA_R_64I      = 24,
  CUDA_C_64I      = 25,
  CUDA_R_64U      = 26,
  CUDA_C_64U      = 27,
  CUDA_R_8F_E4M3  = 28,  // nv_fp8_e4m3
  CUDA_R_8F_UE4M3 = CUDA_R_8F_E4M3,
  CUDA_R_8F_E5M2  = 29,  // nv_fp8_e5m2
  CUDA_R_8F_UE8M0 = 30,  // nv_fp8_e8m0 (exponent-only)
  CUDA_R_6F_E2M3  = 31,  // nv_fp6_e2m3
  CUDA_R_6F_E3M2  = 32,  // nv_fp6_e3m2
  CUDA_R_4F_E2M1  = 33   // nv_fp4_e2m1
} cudaDataType_t;

typedef cudaDataType_t cudaDataType;

/* libraryPropertyType — cuDNN's cudnn_graph.h (real headers) references it. */
typedef enum libraryPropertyType_t {
  MAJOR_VERSION,
  MINOR_VERSION,
  PATCH_LEVEL
} libraryPropertyType;

#endif // __LIBRARY_TYPES_H__
