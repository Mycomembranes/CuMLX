// cufftXt.h — minimal macOS arm64e LINK-STUB header for MLX's CUDA backend.
//
// MLX's fft.cu does `#include <cufftXt.h>`. The cuFFT-Xt extension header
// pulls in the base cufft.h, then adds the cufftXt* entry points MLX uses.
// Pairs with libcufft.dylib (honest no-op: every fn returns
// CUFFT_NOT_IMPLEMENTED). No FFT compute on Apple Silicon — out of scope.
#ifndef MLX_STUB_CUFFTXT_H
#define MLX_STUB_CUFFTXT_H

#include "cufft.h"
#include "library_types.h" // cudaDataType

#ifdef __cplusplus
extern "C" {
#endif

// cufftXtMakePlanMany takes long-long dims/strides and cudaDataType for the
// I/O + execution types (see fft.cu).
cufftResult cufftXtMakePlanMany(
    cufftHandle plan,
    int rank,
    long long int* n,
    long long int* inembed,
    long long int istride,
    long long int idist,
    cudaDataType inputtype,
    long long int* onembed,
    long long int ostride,
    long long int odist,
    cudaDataType outputtype,
    long long int batch,
    size_t* workSize,
    cudaDataType executiontype);

// In/out-of-place execute for an Xt plan (direction = CUFFT_FORWARD/INVERSE).
cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLX_STUB_CUFFTXT_H
