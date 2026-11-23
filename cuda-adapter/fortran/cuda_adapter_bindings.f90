! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0
!
! cuda_adapter_bindings.f90 - ISO_C_BINDING interfaces to the CUDA Driver API
! as exported by the adapter's stand-in libcuda (arm64 slice).
!
! Device pointers are 64-bit integers; handles are opaque C pointers. Large
! data belongs in ALLOCATABLE arrays, never in fixed-size static arrays: big
! static storage inflates the __DATA segment past what dyld will map.
module cuda_adapter_bindings
  use, intrinsic :: iso_c_binding
  implicit none
  private

  integer(c_int), parameter, public :: CUDA_SUCCESS = 0
  integer, parameter, public :: cu_devptr = c_long_long

  public :: cuInit, cuDriverGetVersion, cuDeviceGetCount, cuDeviceGet, cuDeviceGetName
  public :: cuCtxCreate, cuCtxDestroy, cuCtxSynchronize
  public :: cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH
  public :: cuModuleLoad, cuModuleUnload, cuModuleGetFunction, cuLaunchKernel
  public :: cuGetErrorName

  interface
    integer(c_int) function cuInit(flags) bind(C, name="cuInit")
      import :: c_int
      integer(c_int), value :: flags
    end function

    integer(c_int) function cuDriverGetVersion(version) bind(C, name="cuDriverGetVersion")
      import :: c_int
      integer(c_int), intent(out) :: version
    end function

    integer(c_int) function cuDeviceGetCount(count) bind(C, name="cuDeviceGetCount")
      import :: c_int
      integer(c_int), intent(out) :: count
    end function

    integer(c_int) function cuDeviceGet(device, ordinal) bind(C, name="cuDeviceGet")
      import :: c_int
      integer(c_int), intent(out) :: device
      integer(c_int), value :: ordinal
    end function

    integer(c_int) function cuDeviceGetName(name, length, device) bind(C, name="cuDeviceGetName")
      import :: c_int, c_char
      character(kind=c_char), intent(out) :: name(*)
      integer(c_int), value :: length, device
    end function

    integer(c_int) function cuCtxCreate(ctx, flags, device) bind(C, name="cuCtxCreate_v2")
      import :: c_int, c_ptr
      type(c_ptr), intent(out) :: ctx
      integer(c_int), value :: flags, device
    end function

    integer(c_int) function cuCtxDestroy(ctx) bind(C, name="cuCtxDestroy_v2")
      import :: c_int, c_ptr
      type(c_ptr), value :: ctx
    end function

    integer(c_int) function cuCtxSynchronize() bind(C, name="cuCtxSynchronize")
      import :: c_int
    end function

    integer(c_int) function cuMemAlloc(dptr, bytes) bind(C, name="cuMemAlloc_v2")
      import :: c_int, c_long_long, c_size_t
      integer(c_long_long), intent(out) :: dptr
      integer(c_size_t), value :: bytes
    end function

    integer(c_int) function cuMemFree(dptr) bind(C, name="cuMemFree_v2")
      import :: c_int, c_long_long
      integer(c_long_long), value :: dptr
    end function

    integer(c_int) function cuMemcpyHtoD(dst, src, bytes) bind(C, name="cuMemcpyHtoD_v2")
      import :: c_int, c_long_long, c_ptr, c_size_t
      integer(c_long_long), value :: dst
      type(c_ptr), value :: src
      integer(c_size_t), value :: bytes
    end function

    integer(c_int) function cuMemcpyDtoH(dst, src, bytes) bind(C, name="cuMemcpyDtoH_v2")
      import :: c_int, c_long_long, c_ptr, c_size_t
      type(c_ptr), value :: dst
      integer(c_long_long), value :: src
      integer(c_size_t), value :: bytes
    end function

    integer(c_int) function cuModuleLoad(module, path) bind(C, name="cuModuleLoad")
      import :: c_int, c_ptr, c_char
      type(c_ptr), intent(out) :: module
      character(kind=c_char), intent(in) :: path(*)
    end function

    integer(c_int) function cuModuleUnload(module) bind(C, name="cuModuleUnload")
      import :: c_int, c_ptr
      type(c_ptr), value :: module
    end function

    integer(c_int) function cuModuleGetFunction(func, module, name) bind(C, name="cuModuleGetFunction")
      import :: c_int, c_ptr, c_char
      type(c_ptr), intent(out) :: func
      type(c_ptr), value :: module
      character(kind=c_char), intent(in) :: name(*)
    end function

    integer(c_int) function cuLaunchKernel(func, gx, gy, gz, bx, by, bz, shared, stream, params, extra) &
        bind(C, name="cuLaunchKernel")
      import :: c_int, c_ptr
      type(c_ptr), value :: func
      integer(c_int), value :: gx, gy, gz, bx, by, bz, shared
      type(c_ptr), value :: stream
      type(c_ptr), intent(in) :: params(*)
      type(c_ptr), value :: extra
    end function

    integer(c_int) function cuGetErrorName(code, text) bind(C, name="cuGetErrorName")
      import :: c_int, c_ptr
      integer(c_int), value :: code
      type(c_ptr), intent(out) :: text
    end function
  end interface
end module cuda_adapter_bindings
