! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

program nvidiactl_core_control_probe
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int), parameter :: cuda_success = 0
  integer(c_int32_t), parameter :: ioctl_query_pytorch_ready = int(z'102', c_int32_t)
  integer(c_int), parameter :: reason_bytes = 128

  type, bind(C) :: nvidiactl_status
    integer(c_int32_t) :: size
    integer(c_int32_t) :: abi_version
    integer(c_int16_t) :: pci_vendor
    integer(c_int16_t) :: pci_device
    integer(c_int32_t) :: chip_id
    integer(c_int64_t) :: feature_bits
    integer(c_int32_t) :: max_selector
    integer(c_int32_t) :: hardware_ready
    integer(c_int32_t) :: hardware_compute
    integer(c_int32_t) :: pytorch_ready
    integer(c_int32_t) :: claim_level
    integer(c_int8_t) :: reason(reason_bytes)
  end type nvidiactl_status

  interface
    function cudaDriverKitNvidiaCtlOpen(handle) bind(C, name="cudaDriverKitNvidiaCtlOpen")
      import :: c_int, c_int64_t
      integer(c_int64_t) :: handle
      integer(c_int) :: cudaDriverKitNvidiaCtlOpen
    end function

    function cudaDriverKitNvidiaCtlClose(handle) bind(C, name="cudaDriverKitNvidiaCtlClose")
      import :: c_int, c_int64_t
      integer(c_int64_t), value :: handle
      integer(c_int) :: cudaDriverKitNvidiaCtlClose
    end function

    function cudaDriverKitNvidiaCtlIoctl(handle, request, input, input_len, output, output_len, bytes_written) bind(C, name="cudaDriverKitNvidiaCtlIoctl")
      import :: c_int, c_int32_t, c_int64_t, c_ptr, c_size_t
      integer(c_int64_t), value :: handle
      integer(c_int32_t), value :: request
      type(c_ptr), value :: input
      integer(c_size_t), value :: input_len
      type(c_ptr), value :: output
      integer(c_size_t), value :: output_len
      integer(c_size_t) :: bytes_written
      integer(c_int) :: cudaDriverKitNvidiaCtlIoctl
    end function
  end interface

  type(nvidiactl_status), target :: status
  integer(c_int64_t) :: handle
  integer(c_size_t) :: bytes_written
  integer(c_int) :: rc

  rc = cudaDriverKitNvidiaCtlOpen(handle)
  if (rc /= cuda_success) stop 1

  rc = cudaDriverKitNvidiaCtlIoctl(handle, ioctl_query_pytorch_ready, c_null_ptr, 0_c_size_t, c_loc(status), c_sizeof(status), bytes_written)
  if (rc /= cuda_success) stop 2

  print *, "core control probe bytes=", bytes_written, " pytorch_ready=", status%pytorch_ready
  rc = cudaDriverKitNvidiaCtlClose(handle)
  if (rc /= cuda_success) stop 3
end program nvidiactl_core_control_probe
