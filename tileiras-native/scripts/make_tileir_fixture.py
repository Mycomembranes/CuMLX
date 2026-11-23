#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified for macOS ARM64: write a Tile IR bytecode fixture without Linux.
#
# NVIDIA's `cuda-tile` wheel (PyPI, Apache-2.0) ships only manylinux/win builds
# because its front end needs a C extension, but its bytecode WRITER
# (cuda/tile/_bytecode/*.py) is pure Python. This script imports only that
# sub-package from an UNPACKED wheel (never installed, `cuda/tile/__init__.py`
# and the C extension are never executed) and emits a small kernel by calling
# the op encoders directly. Nothing from the wheel is copied into the repo.
#
#   add_f32:     entry(a, b, c: ptr<f32>)            *c = *a + *b
#   vecadd_f32:  entry(a, b, c: ptr<f32>, n: i64)    c[i] = a[i] + b[i] (16-element tiles)
#
# Usage: make_tileir_fixture.py --cuda-tile-root <unpacked wheel> --out FILE
import argparse
import hashlib
import importlib
import os
import sys
import types

WHEEL_SHA256 = "530232e8080c4d831e433d2e50bda0b8e4a2d5f803d9c1a878191852babd2752"  # cuda_tile-1.8.0-cp312-cp312-manylinux2014_aarch64.whl


def load_bytecode_package(root):
    pkg = os.path.join(root, "cuda", "tile")
    if not os.path.isfile(os.path.join(pkg, "_bytecode", "writer.py")):
        raise SystemExit(f"{root} is not an unpacked cuda-tile wheel")
    # Namespace stubs: `cuda` and `cuda.tile` exist, their __init__ never runs.
    for name, path in (("cuda", os.path.join(root, "cuda")), ("cuda.tile", pkg)):
        module = types.ModuleType(name)
        module.__path__ = [path]
        sys.modules[name] = module
    return importlib.import_module("cuda.tile._bytecode")


def add_f32(bc, version):
    buf = bytearray()
    with bc.write_bytecode(num_functions=1, buf=buf, version=version) as writer:
        types_ = writer.type_table
        f32 = types_.F32
        scalar = types_.tile(f32, [])
        pointer = types_.tile(types_.pointer(f32, bc.PtrAttr.Missing), [])
        token = types_.Token
        with writer.function(name="add_f32", parameter_types=[pointer] * 3, result_types=[],
                             entry_point=True, hints={},
                             debug_attr=bc.MISSING_DEBUG_ATTR_ID) as (code, (a, b, c)):
            weak = bc.MemoryOrderingSemantics.WEAK
            x, _ = bc.encode_LoadPtrTkoOp(code, scalar, token, a, None, None, None, weak, None, None)
            y, _ = bc.encode_LoadPtrTkoOp(code, scalar, token, b, None, None, None, weak, None, None)
            total = bc.encode_AddFOp(code, scalar, x, y, bc.RoundingMode.NEAREST_EVEN, False)
            bc.encode_StorePtrTkoOp(code, token, c, total, None, None, weak, None, None)
            bc.encode_ReturnOp(code, [])
    return bytes(buf)


def vecadd_f32(bc, version, tile=16):
    """entry(a, b, c: ptr<f32>, n: i64): c[i] = a[i] + b[i], one 16-element tile per block."""
    buf = bytearray()
    with bc.write_bytecode(num_functions=1, buf=buf, version=version) as writer:
        types_ = writer.type_table
        f32 = types_.F32
        pointer = types_.tile(types_.pointer(f32, bc.PtrAttr.Missing), [])
        length = types_.tile(types_.I64, [])
        index = types_.tile(types_.I32, [])
        block = types_.tile(f32, [tile])
        view = types_.tensor_view(f32, [bc.DYNAMIC_SHAPE], [1], bc.PtrAttr.Missing)
        partition = types_.partition_view([tile], view, [0], bc.PaddingValue.Missing)
        token = types_.Token
        with writer.function(name="vecadd_f32", parameter_types=[pointer] * 3 + [length], result_types=[],
                             entry_point=True, hints={},
                             debug_attr=bc.MISSING_DEBUG_ATTR_ID) as (code, (a, b, c, n)):
            weak = bc.MemoryOrderingSemantics.WEAK
            parts = [bc.encode_MakePartitionViewOp(code, partition,
                                                   bc.encode_MakeTensorViewOp(code, view, p, [n], []))
                     for p in (a, b, c)]
            x_id, _, _ = bc.encode_GetTileBlockIdOp(code, index, index, index)
            x, _ = bc.encode_LoadViewTkoOp(code, block, token, parts[0], [x_id], None, weak, None, None, [])
            y, _ = bc.encode_LoadViewTkoOp(code, block, token, parts[1], [x_id], None, weak, None, None, [])
            total = bc.encode_AddFOp(code, block, x, y, bc.RoundingMode.NEAREST_EVEN, False)
            bc.encode_StoreViewTkoOp(code, token, total, parts[2], [x_id], None, weak, None, None, [])
            bc.encode_ReturnOp(code, [])
    return bytes(buf)


KERNELS = {"add_f32": add_f32, "vecadd_f32": vecadd_f32}


def main():
    ap = argparse.ArgumentParser(description="emit a Tile IR bytecode fixture (pure-Python writer)")
    ap.add_argument("--cuda-tile-root", required=True, help="directory of the UNPACKED cuda-tile wheel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", choices=sorted(KERNELS), default="add_f32")
    ap.add_argument("--bytecode-version", default="V_13_2", help="BytecodeVersion member (tileiras 13.2 reads V_13_2)")
    a = ap.parse_args()
    bc = load_bytecode_package(a.cuda_tile_root)
    data = KERNELS[a.kernel](bc, bc.BytecodeVersion[a.bytecode_version])
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"{a.out}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
