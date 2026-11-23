! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0

! nvgpu-smi (Fortran face) — prints an nvidia-smi-style banner using the
! cuda-driverkit C ABI. Runs WITHOUT silicon: cuDriverGetVersion returns the
! compiled-in constant (12080); cuInit connects best-effort and continues;
! cuDeviceGetCount reports the configured device count.
program nvgpu_smi_fortran
  use, intrinsic :: iso_c_binding
  use cuda_driverkit_bindings
  implicit none

  integer(c_int) :: rc_init, rc_ver, rc_count
  integer(c_int), target :: version, ndev
  integer :: major, minor

  version = 0_c_int
  ndev = 0_c_int

  ! cuInit is best-effort: a nonzero rc (no daemon / no silicon) is non-fatal.
  rc_init = cuInit(0_c_int)

  ! cuDriverGetVersion is silicon-free — always available.
  rc_ver = cuDriverGetVersion(version)

  ! cuDeviceGetCount reports the configured device count (silicon-free stub-safe).
  rc_count = cuDeviceGetCount(ndev)

  ! Decode the packed integer version: major = v/1000, minor = (v/10) mod 100.
  major = version / 1000
  minor = mod(version / 10, 100)

  if (rc_ver /= cuda_success) then
    print '(A,I0,A)', "NVGPU-SMI (Fortran)  cuDriverGetVersion failed (rc=", rc_ver, ")"
    stop 1
  end if

  if (rc_count /= cuda_success) then
    ! Device enumeration unavailable (no daemon/silicon) — still report version.
    print '(A,I0,A,I0,A,I0,A,I0,A)', &
      "NVGPU-SMI (Fortran)  CUDA Driver Version: ", major, ".", minor, &
      " (raw ", version, ")  Devices: unavailable (rc=", rc_count, ")"
  else
    print '(A,I0,A,I0,A,I0,A,I0)', &
      "NVGPU-SMI (Fortran)  CUDA Driver Version: ", major, ".", minor, &
      " (raw ", version, ")  Devices: ", ndev
  end if

  if (rc_init /= cuda_success) then
    print '(A,I0,A)', &
      "  note: cuInit returned ", rc_init, " (no daemon/silicon — version banner is authoritative)"
  end if
end program nvgpu_smi_fortran
