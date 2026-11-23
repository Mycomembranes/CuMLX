#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# run_cuda_rosetta.sh - run a x86_64 CUDA Driver API application through the adapter.
# usage: run_cuda_rosetta.sh <application> [arguments...]
# See run_common.sh for what is checked and the environment it reads.
WANT_ARCH=x86_64
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
