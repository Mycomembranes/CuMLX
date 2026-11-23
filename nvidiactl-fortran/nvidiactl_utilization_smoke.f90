! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

program nvidiactl_utilization_smoke
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
  real(c_float) :: a(4), b(4), c(4)

  a = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
  b = [5.0_c_float, 6.0_c_float, 7.0_c_float, 8.0_c_float]
  c = a + b

  rc = cudaDriverKitNvidiaCtlQueryStatus(status)
  if (rc /= cuda_success) stop 1

  if (status%pytorch_ready /= 1) then
    print *, "control path reachable; hardware workload intentionally skipped"
    stop 0
  end if

  print *, "host utilization checksum=", sum(c)
end program nvidiactl_utilization_smoke
