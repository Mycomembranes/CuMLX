! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

module cuda_driverkit_bindings
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int), parameter :: cuda_success = 0

  interface
    function cuInit(flags) bind(C, name="cuInit")
      import :: c_int
      integer(c_int), value :: flags
      integer(c_int) :: cuInit
    end function

    function cuDeviceGet(device, ordinal) bind(C, name="cuDeviceGet")
      import :: c_int
      integer(c_int) :: device
      integer(c_int), value :: ordinal
      integer(c_int) :: cuDeviceGet
    end function

    function cuDeviceGetCount(count) bind(C, name="cuDeviceGetCount")
      import :: c_int
      integer(c_int) :: count
      integer(c_int) :: cuDeviceGetCount
    end function

    function cuDriverGetVersion(version) bind(C, name="cuDriverGetVersion")
      import :: c_int
      integer(c_int) :: version
      integer(c_int) :: cuDriverGetVersion
    end function

    function cuCtxCreate(pctx, flags, dev) bind(C, name="cuCtxCreate")
      import :: c_int, c_intptr_t
      integer(c_intptr_t) :: pctx
      integer(c_int), value :: flags
      integer(c_int), value :: dev
      integer(c_int) :: cuCtxCreate
    end function

    function cuCtxDestroy(ctx) bind(C, name="cuCtxDestroy")
      import :: c_int, c_intptr_t
      integer(c_intptr_t), value :: ctx
      integer(c_int) :: cuCtxDestroy
    end function

    function cuMemAlloc_v2(dptr, bytesize) bind(C, name="cuMemAlloc_v2")
      import :: c_int, c_int64_t, c_size_t
      integer(c_int64_t) :: dptr
      integer(c_size_t), value :: bytesize
      integer(c_int) :: cuMemAlloc_v2
    end function

    function cuMemFree_v2(dptr) bind(C, name="cuMemFree_v2")
      import :: c_int, c_int64_t
      integer(c_int64_t), value :: dptr
      integer(c_int) :: cuMemFree_v2
    end function

    function cuMemcpyHtoD(dstDevice, srcHost, bytes) bind(C, name="cuMemcpyHtoD")
      import :: c_int, c_int64_t, c_ptr, c_size_t
      integer(c_int64_t), value :: dstDevice
      type(c_ptr), value :: srcHost
      integer(c_size_t), value :: bytes
      integer(c_int) :: cuMemcpyHtoD
    end function

    function cuMemcpyDtoH(dstHost, srcDevice, bytes) bind(C, name="cuMemcpyDtoH")
      import :: c_int, c_int64_t, c_ptr, c_size_t
      type(c_ptr), value :: dstHost
      integer(c_int64_t), value :: srcDevice
      integer(c_size_t), value :: bytes
      integer(c_int) :: cuMemcpyDtoH
    end function

    function cuModuleLoad(module, fname) bind(C, name="cuModuleLoad")
      import :: c_char, c_int, c_intptr_t
      integer(c_intptr_t) :: module
      character(kind=c_char) :: fname(*)
      integer(c_int) :: cuModuleLoad
    end function

    function cuModuleGetFunction(hfunc, hmod, name) bind(C, name="cuModuleGetFunction")
      import :: c_char, c_int, c_intptr_t
      integer(c_intptr_t) :: hfunc
      integer(c_intptr_t), value :: hmod
      character(kind=c_char) :: name(*)
      integer(c_int) :: cuModuleGetFunction
    end function

    function cuModuleUnload(hmod) bind(C, name="cuModuleUnload")
      import :: c_int, c_intptr_t
      integer(c_intptr_t), value :: hmod
      integer(c_int) :: cuModuleUnload
    end function

    function cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, &
                            blockDimX, blockDimY, blockDimZ, &
                            sharedMemBytes, hStream, kernelParams, extra) &
                            bind(C, name="cuLaunchKernel")
      import :: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: f
      integer(c_int), value :: gridDimX
      integer(c_int), value :: gridDimY
      integer(c_int), value :: gridDimZ
      integer(c_int), value :: blockDimX
      integer(c_int), value :: blockDimY
      integer(c_int), value :: blockDimZ
      integer(c_int), value :: sharedMemBytes
      integer(c_intptr_t), value :: hStream
      type(c_ptr), value :: kernelParams
      type(c_ptr), value :: extra
      integer(c_int) :: cuLaunchKernel
    end function

    function cuCtxSynchronize() bind(C, name="cuCtxSynchronize")
      import :: c_int
      integer(c_int) :: cuCtxSynchronize
    end function
  end interface
end module cuda_driverkit_bindings
