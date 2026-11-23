# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host smoke test: the hybrid_ops CUDA backend through the adapter.

Loads ~/lib/hybrid_ops/backends/cuda_backend.py unmodified and points it at
the out-of-tree libcuda_compute.dylib built against the adapter. Nothing in
the hybrid_ops tree is written.

usage: hybrid_ops_smoke.py <libcuda_compute.dylib> [--expect-compute]

Checks, in order: library load and init, device name, a tensor upload and
readback round trip, and pipeline creation (the kernel module's PTX is
compiled by the adapter and the entry point is found). Without
--expect-compute the launch is expected to be refused: the loopback backend
executes no hybrid_ops kernel. With it (a served card), rms_norm is compared
with NumPy.
"""
import importlib.util
import os
import sys

import numpy as np


def main() -> int:
    library = os.path.abspath(sys.argv[1])
    expect_compute = "--expect-compute" in sys.argv[2:]
    backend_path = os.path.expanduser(os.environ.get(
        "HYBRID_OPS_CUDA_BACKEND", "~/lib/hybrid_ops/backends/cuda_backend.py"))
    spec = importlib.util.spec_from_file_location("hybrid_ops_cuda_backend", backend_path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    # Redirect the library search in this process only.
    backend._LIB_SEARCH_PATHS = [library]

    backend._ensure_cuda()
    print("device:", backend.cuda_get_device_name())

    data = np.arange(4096, dtype=np.float32).reshape(64, 64) * 0.5
    tensor = backend.CUDATensor.from_numpy(data)
    back = tensor.numpy()
    if not np.array_equal(back.reshape(data.shape), data):
        print("FAIL: upload/readback round trip differs")
        return 1
    print("round trip: ok (", data.nbytes, "bytes )")

    pipeline = backend._get_or_create_pipeline("rms_norm")
    print("pipeline: ok (handle", pipeline, ")")

    x = np.random.default_rng(0).standard_normal((8, 256)).astype(np.float32)
    weight = np.ones(256, dtype=np.float32)
    try:
        result = np.asarray(backend.rms_norm(x, weight, eps=1e-5))
    except Exception as exc:  # the C library reports a refused launch as an error
        if expect_compute:
            print("FAIL: launch failed:", exc)
            return 1
        print("launch refused as expected on a backend with no hybrid_ops kernels:", str(exc)[:120])
        print("hybrid_ops_smoke: PASS (host path; compute not exercised)")
        return 0
    if not expect_compute:
        print("FAIL: the launch unexpectedly succeeded without a device backend")
        return 1
    want = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * weight
    worst = float(np.max(np.abs(result.reshape(x.shape) - want)))
    print("rms_norm max abs error vs NumPy:", worst)
    if worst > 1e-4:
        print("FAIL: rms_norm differs from NumPy")
        return 1
    print("hybrid_ops_smoke: PASS (compute checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
