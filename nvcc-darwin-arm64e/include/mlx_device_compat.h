// mlx_device_compat.h — Darwin-arm64 ↔ clang-CUDA device-compile compatibility.
//
// Force-included (via `-include`) by the nvcc-darwin shim's cu_to_ptx() when
// compiling a `.cu` for the NVPTX device. It papers over two clang-CUDA-on-macOS
// gaps that a real Linux + nvcc build never sees, WITHOUT patching MLX sources or
// the clang headers:
//
//   * `long double` math overloads. clang's CUDA cmath wrapper
//     (__clang_cuda_cmath.h) only declares isnan/isinf/signbit for float + double.
//     {fmt} instantiates write<long double>, calling std::isnan(long double) →
//     ambiguous (float vs double both rank-equal). nvcc sidesteps this because it
//     treats `long double` as `double`. We add the missing long-double overloads
//     (delegating to the double intrinsic, which is correct since NVPTX has no
//     wider-than-double FP).
//
// The ARM-feature undefs (__ARM_FEATURE_FP16_SCALAR_ARITHMETIC / __ARM_FEATURE_BF16)
// are passed as -U on the command line by the shim, not here, because they must be
// undefined *before* any header (including this one) is parsed.

#pragma once

#if defined(__CUDA_ARCH__)
#include <cmath>
namespace std {
__attribute__((device)) inline bool isnan(long double __x) {
  return ::__isnan(static_cast<double>(__x));
}
__attribute__((device)) inline bool isinf(long double __x) {
  return ::__isinf(static_cast<double>(__x));
}
__attribute__((device)) inline bool signbit(long double __x) {
  return ::__signbitd(static_cast<double>(__x));
}
} // namespace std
#endif // __CUDA_ARCH__

#if !defined(__CUDA_ARCH__)
// HOST side of a .cu compiled with --cuda-host-only (the nvcc-darwin shim's
// linkable-object pass). clang-21's <<<kernel>>> lowering for the partially-
// supported CUDA 12.8 redist emits the LEGACY launch ABI
// (cudaConfigureCall / cudaSetupArgument / cudaLaunch), which CUDA 12 headers no
// longer declare. Pull cuda_runtime.h (for dim3 / cudaStream_t / cudaError_t)
// then declare them so the host pass compiles. They are resolved by the cudart
// shim's stubs at link; kernels are not executed on macOS (sp3 is canonical).
#include <cuda_runtime.h>
extern "C" {
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                              size_t sharedMem, cudaStream_t stream);
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void* func);
}
#endif // !__CUDA_ARCH__
