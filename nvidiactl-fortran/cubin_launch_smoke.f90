! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

program cubin_launch_smoke
  use, intrinsic :: iso_c_binding
  use cuda_driverkit_bindings
  implicit none

  integer(c_int), parameter :: n = 256
  integer(c_size_t), parameter :: nbytes = int(n, c_size_t) * 4_c_size_t

  real(c_float), target :: a(n), b(n), c(n)
  integer(c_int64_t), target :: d_a, d_b, d_c
  integer(c_int), target :: n_arg
  type(c_ptr), target :: kernel_params(4)
  integer(c_intptr_t) :: ctx, mod_h, func_h
  integer(c_int) :: dev, i
  character(len=4096) :: cubin_path
  character(kind=c_char), allocatable, target :: cubin_c(:), func_c(:)
  integer :: path_len, env_status
  logical :: ok

  d_a = 0_c_int64_t
  d_b = 0_c_int64_t
  d_c = 0_c_int64_t
  ctx = 0_c_intptr_t
  mod_h = 0_c_intptr_t
  func_h = 0_c_intptr_t
  dev = 0_c_int
  n_arg = n

  do i = 1, n
    a(i) = real(i, c_float)
    b(i) = real(2 * i, c_float)
    c(i) = 0.0_c_float
  end do

  call get_environment_variable("CUBIN", cubin_path, length=path_len, status=env_status)
  if (env_status /= 0 .or. path_len <= 0) then
    cubin_path = "../../crates/cuda-driverkit/test_assets/cubins/vecadd_macos.cubin"
    path_len = len_trim(cubin_path)
  end if
  call to_c_string(cubin_path, path_len, cubin_c)
  call to_c_string("vecadd_macos", len("vecadd_macos"), func_c)

  call check(cuInit(0_c_int), "cuInit")
  call check(cuDeviceGet(dev, 0_c_int), "cuDeviceGet")
  call check(cuCtxCreate(ctx, 0_c_int, dev), "cuCtxCreate")

  call check(cuMemAlloc_v2(d_a, nbytes), "cuMemAlloc_v2(A)")
  call check(cuMemAlloc_v2(d_b, nbytes), "cuMemAlloc_v2(B)")
  call check(cuMemAlloc_v2(d_c, nbytes), "cuMemAlloc_v2(C)")

  call check(cuMemcpyHtoD(d_a, c_loc(a(1)), nbytes), "cuMemcpyHtoD(A)")
  call check(cuMemcpyHtoD(d_b, c_loc(b(1)), nbytes), "cuMemcpyHtoD(B)")

  call check(cuModuleLoad(mod_h, cubin_c), "cuModuleLoad")
  call check(cuModuleGetFunction(func_h, mod_h, func_c), "cuModuleGetFunction")

  kernel_params(1) = c_loc(d_a)
  kernel_params(2) = c_loc(d_b)
  kernel_params(3) = c_loc(d_c)
  kernel_params(4) = c_loc(n_arg)

  call check(cuLaunchKernel(func_h, 1_c_int, 1_c_int, 1_c_int, &
                            256_c_int, 1_c_int, 1_c_int, &
                            0_c_int, 0_c_intptr_t, c_loc(kernel_params(1)), &
                            c_null_ptr), "cuLaunchKernel")
  call check(cuCtxSynchronize(), "cuCtxSynchronize")
  call check(cuMemcpyDtoH(c_loc(c(1)), d_c, nbytes), "cuMemcpyDtoH(C)")

  ok = .true.
  do i = 1, n
    if (abs(c(i) - (a(i) + b(i))) > 1.0e-5_c_float) then
      print *, "FAIL: mismatch at", i, " expected=", a(i) + b(i), " got=", c(i)
      ok = .false.
      exit
    end if
  end do

  call check(cuModuleUnload(mod_h), "cuModuleUnload")
  call check(cuMemFree_v2(d_c), "cuMemFree_v2(C)")
  call check(cuMemFree_v2(d_b), "cuMemFree_v2(B)")
  call check(cuMemFree_v2(d_a), "cuMemFree_v2(A)")
  call check(cuCtxDestroy(ctx), "cuCtxDestroy")

  if (ok) then
    print *, "PASS: cubin vecadd N=", n
  else
    stop 10
  end if

contains
  subroutine check(rc, name)
    integer(c_int), intent(in) :: rc
    character(len=*), intent(in) :: name

    if (rc /= cuda_success) then
      print *, trim(name), " failed, CUresult=", rc
      stop 1
    end if
  end subroutine check

  subroutine to_c_string(text, text_len, c_text)
    character(len=*), intent(in) :: text
    integer, intent(in) :: text_len
    character(kind=c_char), allocatable, intent(out) :: c_text(:)
    integer :: j

    allocate(character(kind=c_char) :: c_text(text_len + 1))
    do j = 1, text_len
      c_text(j) = text(j:j)
    end do
    c_text(text_len + 1) = c_null_char
  end subroutine to_c_string
end program cubin_launch_smoke
