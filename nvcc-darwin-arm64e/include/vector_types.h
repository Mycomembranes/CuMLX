// vector_types.h — minimal CUDA vector types for the macOS HOST compile.
//
// CCCL's <cuda/std/__tuple_dir/vector_types.h> (enabled by _CCCL_HAS_CTK(),
// i.e. whenever a CUDA toolkit is detected) specializes std::tuple_size/
// tuple_element/get for the CUDA vector types char1..ulonglong4. It does NOT
// define them — it assumes a real <vector_types.h> was pulled first. On macOS
// the device path uses NVIDIA's real header (via --cuda-path); this shim
// supplies the host-side definitions so CCCL's tuple specializations resolve.
//
// uint3 + dim3 are the launch-config types owned by cuda_runtime.h (which
// #includes this header); they are intentionally NOT redefined here.
#pragma once
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__

#define _OXIDE_VEC1(tag, T) struct tag##1 { T x; };
#define _OXIDE_VEC2(tag, T) struct tag##2 { T x; T y; };
#define _OXIDE_VEC3(tag, T) struct tag##3 { T x; T y; T z; };
#define _OXIDE_VEC4(tag, T) struct tag##4 { T x; T y; T z; T w; };
#define _OXIDE_VEC(tag, T) \
  _OXIDE_VEC1(tag, T) _OXIDE_VEC2(tag, T) _OXIDE_VEC3(tag, T) _OXIDE_VEC4(tag, T)

_OXIDE_VEC(char, signed char)
_OXIDE_VEC(uchar, unsigned char)
_OXIDE_VEC(short, short)
_OXIDE_VEC(ushort, unsigned short)
_OXIDE_VEC(int, int)
// uint3 is owned by cuda_runtime.h (launch-config type); define only 1/2/4 here.
_OXIDE_VEC1(uint, unsigned int)
_OXIDE_VEC2(uint, unsigned int)
_OXIDE_VEC4(uint, unsigned int)
_OXIDE_VEC(long, long)
_OXIDE_VEC(ulong, unsigned long)
_OXIDE_VEC(longlong, long long)
_OXIDE_VEC(ulonglong, unsigned long long)
_OXIDE_VEC(float, float)
_OXIDE_VEC(double, double)

#undef _OXIDE_VEC
#undef _OXIDE_VEC1
#undef _OXIDE_VEC2
#undef _OXIDE_VEC3
#undef _OXIDE_VEC4

#endif // __VECTOR_TYPES_H__
