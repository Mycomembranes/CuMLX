# CUDA device adapter (legacy CUDA Driver API clients -> the updated dext)

NVIDIA's last macOS CUDA driver (418.105, `CUDA.framework` 10.1.105) is x86_64
only. Its kext cannot load on Apple Silicon and its library talks to a driver
that does not exist here, so its `cuInit` can never succeed on this Mac.

This adapter replaces the **library boundary** and leaves everything else alone:

```
x86_64 app (Rosetta) ─┐   stand-in libcuda (fat dylib)        arm64                   existing stack
arm64 app / Fortran ──┴── AF_UNIX, cuda-oxide.driver.v1 ──> cuda_adapter_server ──> launch endpoint ──> pci_server_full ──> dext
```

- The application is not modified. NVIDIA's installed files are never written;
  the stand-in is put first on the dyld search paths by the launcher.
- `cuda_adapter_server` never opens DriverKit. Its `driverkit` backend calls the
  existing `cuda-driverkit` C ABI in-process, which is a client of the
  exact-serve launch endpoint. The card keeps one owner (`pci_server_full`).
- PTX is compiled to sm_120 by the repo's verified native `ptxas`
  (`tools/ptxas-resolve.sh`), never a `ptxas` from `PATH`.

Layout follows the DALI x64 -> Apple Silicon port: small patches beside the
tree (`patches/`, none needed yet), one build script, run wrappers, and the
limits stated plainly.

## Build and self-test (host only, no card)

```bash
portkit/cuda-adapter/scripts/build_cuda_adapter.sh            # build + self-test
portkit/cuda-adapter/scripts/build_cuda_adapter.sh --install  # also copy to ~/.config/cuda-oxide/cuda-adapter-root
cargo test -p cuda-adapter-server
```

The self-test starts its own server with the `loopback` backend (host memory,
one emulated reference kernel) on a private socket. It never connects to a
served stack.

## Run a client

The adapter server is an ordinary shared-CUDA client of ONE served card. It is
pointed at that card's registry (`serve-pids.<id>.json`), the same convention
as `CUDA_OXIDE_EXACT_SERVE_STATE` for any other C-ABI client, and reaches the
card only through that stack's launch endpoint. It is not part of the
supervisor and changes nothing in `egpu_serve_exact.py` / `egpu_device_server.py`.

```bash
# 1. attended: the card is served with --shared-cuda and is "ready".
# 2. plan + read-only registry preflight (starts nothing):
scripts/run_adapter_server.sh --serve-state "/abs/path/serve-pids.<id>.json"
# 3. start it in the foreground (this is the step that needs the operator's GO):
scripts/run_adapter_server.sh --serve-state "/abs/path/serve-pids.<id>.json" --go
# 4. clients, in another terminal:
scripts/run_cuda_rosetta.sh /path/to/x86_64_cuda_app [args]
scripts/run_cuda_native.sh  /path/to/arm64_cuda_app  [args]
```

The server authenticates to the stack when it starts and refuses to start if
the registry is not a qualified shared-CUDA stack of this boot. Stop it
(Ctrl-C / SIGTERM) before stopping the stack.

The launchers check the application's architecture, refuse binaries for which
macOS drops `DYLD_*`, and run a readiness probe before handing off. A binary
linked against `/usr/local/cuda/lib/libcuda.dylib` by absolute path is
redirected too (checked: dyld searches `DYLD_LIBRARY_PATH` by leaf name first).

Fortran clients use `fortran/cuda_adapter_bindings.f90` and link `-lcuda`.

## Status

| Piece | State |
|---|---|
| Stand-in library, both slices, 401-symbol export list | built; exports diffed against the installed framework |
| Server, wire protocol, module loader, shared segments | 17 unit tests |
| x86_64 under Rosetta, arm64, Fortran, PTX path | pass against the `loopback` backend |
| `driverkit` backend | exercised end to end on the host: client -> adapter -> real C ABI -> real launch server + shared-CUDA backend -> CPU fake device (`scripts/tests/test_cuda_adapter_shared_driver.py`, arm64 and Rosetta). First card run: see below. |
| Stack integration | done as a client: `scripts/run_adapter_server.sh` (plan, registry preflight, `--go`). No supervisor changes. |
| hybrid_ops client | `build/hybrid_ops/libcuda_compute.dylib`, built from the unmodified source; the unmodified Python backend loads it, moves tensors and builds its pipeline (PTX compiled by the adapter). `rms_norm` checked on the card (see below); the other kernels have not been run. |
| Runtime layer | `libcudart_adapter.dylib`: `cudaMalloc/Free/Memcpy/MallocHost/FreeHost/MemGetInfo`, streams, error text. Not a `libcudart` replacement. |
| Shared host memory, client <-> adapter | done: `cuMemAllocHost` / `cuMemHostAlloc` return a segment the server maps too (descriptor passed with `SCM_RIGHTS`); copies on it carry no socket payload. Checked natively and under Rosetta. |
| Shared host memory, adapter <-> card owner | not built, on purpose: that hop is a local socket (GiB/s) in front of a staging path measured at 50-63 MiB/s, so removing it would not change throughput. The gain is in making the owner's staging buffers themselves shared (pinned pages), which is an attended stage. |
| Dext selectors for releasable pins | not started |

## First run on a card (2026-11-20, desk-egpu, RTX 5060 Ti, shared CUDA, Tier 1, link at Gen1 x4)

Attended, small workloads only; every step passed and nothing stalled.

| Step | Result |
|---|---|
| vecadd, native arm64 and x86_64 under Rosetta | three launch forms each, exact |
| PTX module compiled by the adapter, then launched | exact |
| `cublasLtMatmul` on device pointers, native and Rosetta | bit-exact (37x21 * 21x50) |
| hybrid_ops `rms_norm` through its Python backend | max abs error 4.8e-7 against NumPy |

Not covered: transfers above 1 MiB, `matmul_fp16_mma`, long runs, the second card, Tier 2. One small kernel
from the native `ptxas` running correctly does not qualify that compiler in general.

## Limits

| Limit | Detail |
|---|---|
| Driver API only | 77 of the 401 CUDA 10.1 symbols are implemented; the rest return `CUDA_ERROR_NOT_SUPPORTED`. NVIDIA's x86_64 `libcudart` / cuBLAS need private `cuGetExportTable` tables and do not work. |
| Module content | A module must carry PTX or an sm_120 cubin. A module with only pre-Blackwell cubins is refused with `CUDA_ERROR_NO_BINARY_FOR_GPU`. |
| Fat binaries | The parser follows the layout used by open-source decoders and is tested only on a synthetic image. Compressed entries are refused. |
| Compiler qualification | Cubins from the native `ptxas` are host-verified only; CUDA 10 PTX through it has not run on silicon. |
| Launch | Default stream, kernels of at most 64 parameters. The shared-CUDA backend additionally refuses dynamic shared memory and parameters wider than 8 bytes. |
| Streams | A stream is an identity only: every call returns after the work is done. |
| Events | `cuEvent*` record the **host** clock. An elapsed time includes the socket round trip and is not GPU time. |
| Interior copies | The served stack copies only from the start of an allocation. A copy at `base + offset` is emulated with prefix copies (read: download `[0, offset+len)`; write: download `[0, offset)`, then upload), so it costs a transfer proportional to the offset. Kernel pointer arguments at an offset are not affected. |
| Memory totals | `cuMemGetInfo` / `cuDeviceTotalMem` return `CUDA_ERROR_NOT_SUPPORTED` on the served stack, which does not report them. |
| Clients | One client at a time; a second connection is refused as busy. Context and handle state in the library is process-wide, not per-thread. |
| Transfers | 768 MiB per copy. Copies from ordinary host memory travel over the socket in 16 MiB chunks; copies on `cuMemAllocHost` memory do not. `cuMemHostRegister` of existing memory is not supported. |
| Host segments | At most 64 per client, 768 MiB each, sizes rounded up to 16 KiB. `CUDA_ADAPTER_NO_SHARED_HOST=1` forces ordinary memory. |
| x86_64 un-suffixed names | `cuMemAlloc`, `cuMemcpyHtoD`, ... keep the CUDA 3.x ABI (32-bit device pointers), as in the real 10.1 library; they fail if the device address is above 4 GiB. The `_v2` names, which the 10.1 header selects, are unaffected. On arm64 un-suffixed names are the 64-bit ABI. |
| Rosetta | Client applications only. The toolchain ban in `tools/ptxas-resolve.sh` stands. |
| Hardened binaries | macOS ignores `DYLD_*` for hardened-runtime and system binaries; the launcher refuses them. |

## hybrid_ops

`~/lib/drivers/cuda/cuda_compute.c` and `~/lib/hybrid_ops/backends/cuda_backend.py`
disagree on five functions (device name, last error, the two dispatch calls,
synchronize), so the Python backend cannot use the C library as written.
Neither file is edited: the build compiles the C source unchanged with the four
clashing names renamed and adds `shim/hybrid_ops_wrapper_abi.c`, which presents
the interface the Python backend declares. Its one assumption is a block of
(256, 1, 1), taken from the Python callers and `kernels/cuda_kernels.h`.

The Python backend looks for the library only at
`~/lib/drivers/cuda/build/libcuda_compute.dylib`. To use the adapter build
from that tree (this writes one symlink into it, so it is left to the operator):

```bash
mkdir -p ~/lib/drivers/cuda/build
ln -s "$PWD/portkit/cuda-adapter/build/hybrid_ops/libcuda_compute.dylib" ~/lib/drivers/cuda/build/
```

`tests/hybrid_ops_smoke.py <library> [--expect-compute]` runs the backend
without that symlink; with `--expect-compute` (a served card) it compares
`rms_norm` with NumPy.

## Sizing an application's API surface

```bash
CUDA_ADAPTER_TRACE=1 CUDA_ADAPTER_TRACE_FILE=/tmp/calls.log scripts/run_cuda_rosetta.sh ./app
sort /tmp/calls.log | uniq -c | sort -rn
```

`CUDA_ADAPTER_TRACE_PERMISSIVE=1` makes unsupported calls return success so the
application keeps going and the whole surface is listed. Their outputs are not
filled in, so use it for sizing only, never for a real run.

## Safety

The adapter starts no server on its own and drives no card. Running the
`driverkit` backend is a silicon run: it needs a served stack, the host-safety
gates and a fresh operator GO, like every other drive. The server is never
killed mid-call; on SIGTERM it stops accepting and lets the request in flight
finish.
