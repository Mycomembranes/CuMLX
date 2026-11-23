! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

program nvidiactl_abi_smoke
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int), parameter :: cuda_success = 0
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
    function cudaDriverKitNvidiaCtlQueryStatus(status) bind(C, name="cudaDriverKitNvidiaCtlQueryStatus")
      import :: c_int, nvidiactl_status
      type(nvidiactl_status) :: status
      integer(c_int) :: cudaDriverKitNvidiaCtlQueryStatus
    end function
  end interface

  type(nvidiactl_status) :: status
  integer(c_int) :: rc

  rc = cudaDriverKitNvidiaCtlQueryStatus(status)
  if (rc /= cuda_success) then
    print *, "nvidiactl ABI query failed, CUresult=", rc
    stop 1
  end if

  if (status%abi_version /= 1) then
    print *, "unexpected ABI version", status%abi_version
    stop 2
  end if

  print *, "nvidiactl ABI ok; pytorch_ready=", status%pytorch_ready
end program nvidiactl_abi_smoke
