! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0
!
! fortran_vecadd.f90 - vecadd through the adapter from Fortran.
! usage: fortran_vecadd <vecadd cubin or PTX file> [kernel name]
program fortran_vecadd
  use, intrinsic :: iso_c_binding
  use cuda_adapter_bindings
  implicit none

  integer, parameter :: n = 4096
  real(c_float), allocatable, target :: a(:), b(:), c(:)
  integer(cu_devptr), target :: da, db, dc
  integer(c_int), target :: count
  integer(c_int) :: device, version, ndev, i, bad
  type(c_ptr) :: ctx, module, func
  type(c_ptr) :: params(4)
  character(len=1024) :: path, kernel
  character(kind=c_char) :: name(128)
  integer(c_size_t) :: bytes

  if (command_argument_count() < 1) then
    write (*, '(a)') 'usage: fortran_vecadd <vecadd cubin or PTX file> [kernel name]'
    stop 2
  end if
  call get_command_argument(1, path)
  kernel = 'vecadd_macos'
  if (command_argument_count() >= 2) call get_command_argument(2, kernel)

  allocate (a(n), b(n), c(n))
  do i = 1, n
    a(i) = real(i, c_float) * 0.25_c_float
    b(i) = real(n - i, c_float)
  end do
  c = 0.0_c_float
  bytes = int(n, c_size_t) * c_sizeof(a(1))

  call check(cuInit(0_c_int), 'cuInit')
  call check(cuDriverGetVersion(version), 'cuDriverGetVersion')
  call check(cuDeviceGetCount(ndev), 'cuDeviceGetCount')
  call check(cuDeviceGet(device, 0_c_int), 'cuDeviceGet')
  call check(cuDeviceGetName(name, int(size(name), c_int), device), 'cuDeviceGetName')
  call check(cuCtxCreate(ctx, 0_c_int, device), 'cuCtxCreate')
  call check(cuModuleLoad(module, trim(path)//c_null_char), 'cuModuleLoad')
  call check(cuModuleGetFunction(func, module, trim(kernel)//c_null_char), 'cuModuleGetFunction')
  call check(cuMemAlloc(da, bytes), 'cuMemAlloc a')
  call check(cuMemAlloc(db, bytes), 'cuMemAlloc b')
  call check(cuMemAlloc(dc, bytes), 'cuMemAlloc c')
  call check(cuMemcpyHtoD(da, c_loc(a), bytes), 'cuMemcpyHtoD a')
  call check(cuMemcpyHtoD(db, c_loc(b), bytes), 'cuMemcpyHtoD b')

  count = n
  params = [c_loc(da), c_loc(db), c_loc(dc), c_loc(count)]
  call check(cuLaunchKernel(func, int((n + 255) / 256, c_int), 1_c_int, 1_c_int, 256_c_int, 1_c_int, 1_c_int, &
                            0_c_int, c_null_ptr, params, c_null_ptr), 'cuLaunchKernel')
  call check(cuCtxSynchronize(), 'cuCtxSynchronize')
  call check(cuMemcpyDtoH(c_loc(c), dc, bytes), 'cuMemcpyDtoH c')

  bad = 0
  do i = 1, n
    if (c(i) /= a(i) + b(i)) bad = bad + 1
  end do

  call check(cuMemFree(da), 'cuMemFree a')
  call check(cuMemFree(db), 'cuMemFree b')
  call check(cuMemFree(dc), 'cuMemFree c')
  call check(cuModuleUnload(module), 'cuModuleUnload')
  call check(cuCtxDestroy(ctx), 'cuCtxDestroy')

  if (bad /= 0) then
    write (*, '(a,i0,a)') 'fortran_vecadd: FAILED (', bad, ' wrong elements)'
    stop 1
  end if
  write (*, '(a,i0,a)') 'fortran_vecadd: PASS (', n, ' elements)'

contains

  subroutine check(code, what)
    integer(c_int), intent(in) :: code
    character(len=*), intent(in) :: what
    if (code /= CUDA_SUCCESS) then
      write (*, '(a,a,a,i0)') 'fortran_vecadd: ', what, ' failed with CUDA error ', code
      stop 1
    end if
  end subroutine check

end program fortran_vecadd
