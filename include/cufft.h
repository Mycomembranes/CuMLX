// cufft.h — minimal macOS arm64e LINK-STUB header for MLX's CUDA backend.
//
// Declares ONLY the cuFFT surface MLX (fft.cu) references. Pairs with
// libcufft.dylib (honest no-op: every fn returns CUFFT_NOT_IMPLEMENTED). There
// is no FFT compute on Apple Silicon here — out of scope.
//
// MLX includes <cufftXt.h>; cufftXt.h forwards to this header (see cufftXt.h).
#ifndef MLX_STUB_CUFFT_H
#define MLX_STUB_CUFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// cufftHandle is an int in cuFFT (NOT a pointer).
typedef int cufftHandle;

// --- result enum (only the values MLX compares against) ---
typedef enum cufftResult_t {
  CUFFT_SUCCESS = 0,
  CUFFT_NOT_IMPLEMENTED = 14
} cufftResult;

// FFT direction constants MLX passes to cufftXtExec.
#define CUFFT_FORWARD (-1)
#define CUFFT_INVERSE (1)

typedef struct CUstream_st* cudaStream_t;

// --- functions MLX calls (all stubbed to CUFFT_NOT_IMPLEMENTED) ---
cufftResult cufftCreate(cufftHandle* plan);
cufftResult cufftDestroy(cufftHandle plan);
cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);
cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLX_STUB_CUFFT_H
