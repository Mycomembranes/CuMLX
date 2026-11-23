# nvidiactl Fortran Port Kit

This directory contains the three Fortran experiments for the macOS
`/dev/nvidiactl` compatibility layer:

- `nvidiactl_abi_smoke.f90` validates the C ABI layout through
  `ISO_C_BINDING`.
- `nvidiactl_utilization_smoke.f90` is the placeholder for numerics/workload
  probes above the control layer.
- `nvidiactl_core_control_probe.f90` intentionally exercises the control-path
  ABI directly, but remains outside the default driver core.

Build with Homebrew `gfortran`:

```sh
make CUDA_DRIVERKIT_LIB=/path/to/libcuda_driverkit.dylib
```

The Makefile uses the same DALI-style runtime policy:

```text
-O3 -static-libgfortran -static-libgcc
```

These probes must not be used as PyTorch-readiness proof by themselves. The C
ABI reports `pytorch_ready=1` only after the existing hardware completion-fence
gate is satisfied.

## ABI v1 Boundary

`CudaDriverKitNvidiaCtlStatus` / `nvidiactl_status` is intentionally fixed at
ABI v1 for the Fortran probes. Do not add boot telemetry fields to that struct
without defining an ABI v2 and updating every language binding together.

The 2026-06-07 low32 held-session INIT evidence, including the promising
`SEC2_MAILBOX0=0xedd97000` and `SEC2_MAILBOX1=0x74000002` pair, is exposed
through daemon JSON/msgpack as `init_done_wait_sec2_mailbox0/1`, not through the
Fortran nvctl status struct.
