// nvrtc.h — minimal macOS arm64e LINK-STUB header for MLX's CUDA backend.
//
// Declares ONLY the NVRTC surface MLX (jit_module.cpp) references. Pairs with
// libnvrtc.dylib (honest no-op: every fn returns an NVRTC_ERROR_*). There is
// no in-process CUDA-C compiler on Apple Silicon; an offline route exists via
// tools/nvcc-darwin-arm64e / the cuda-oxide PTX pipeline, but wiring it is out
// of scope — this is link-completeness only.
#ifndef MLX_STUB_NVRTC_H
#define MLX_STUB_NVRTC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- result enum (only the values MLX compares against) ---
typedef enum {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;

// --- opaque program handle ---
typedef struct _nvrtcProgram* nvrtcProgram;

// --- functions MLX calls (all stubbed to NVRTC_ERROR_*, except
//     nvrtcVersion which honestly reports 12.8) ---
const char* nvrtcGetErrorString(nvrtcResult result);
nvrtcResult nvrtcVersion(int* major, int* minor);

nvrtcResult nvrtcCreateProgram(
    nvrtcProgram* prog,
    const char* src,
    const char* name,
    int numHeaders,
    const char* const* headers,
    const char* const* includeNames);
nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog);

nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* nameExpression);
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options);

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log);

nvrtcResult nvrtcGetLoweredName(
    nvrtcProgram prog,
    const char* nameExpression,
    const char** loweredName);

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet);
nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MLX_STUB_NVRTC_H
