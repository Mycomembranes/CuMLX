// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device adapter server.
//!
//! Serves the classic CUDA Driver API over the `cuda-oxide.driver.v1` wire
//! (4-byte big-endian length + msgpack `{op, seq, args}`) on a private AF_UNIX
//! socket. Clients are the stand-in `libcuda` forwarders built from
//! `portkit/cuda-adapter/shim/cuda_fwd.c` (x86_64 under Rosetta, or arm64).
//!
//! The server never opens DriverKit. Its `driverkit` backend calls the existing
//! `cuda-driverkit` C ABI in-process, which reaches the card through the
//! exact-serve launch endpoint. The `loopback` backend is a host-only test
//! double: it keeps buffers in host memory and emulates one reference kernel.

pub mod backend;

pub mod backend_driverkit;
pub mod module;
pub mod proto;
pub mod server;
pub mod shm;

/// Wire protocol name reported by `hello`.
pub const PROTOCOL: &str = "cuda-oxide.driver.v1";
/// `server_kind` reported by `hello`.
pub const SERVER_KIND: &str = "cuda-adapter-server";

/// CUDA result codes used on the wire (values from the CUDA 10.1 `cuda.h`).
pub mod cu {
    pub const SUCCESS: i32 = 0;
    pub const INVALID_VALUE: i32 = 1;
    pub const OUT_OF_MEMORY: i32 = 2;
    pub const NOT_INITIALIZED: i32 = 3;
    pub const INVALID_DEVICE: i32 = 101;
    pub const INVALID_IMAGE: i32 = 200;
    pub const INVALID_CONTEXT: i32 = 201;
    pub const NO_BINARY_FOR_GPU: i32 = 209;
    pub const INVALID_HANDLE: i32 = 400;
    pub const NOT_FOUND: i32 = 500;
    pub const LAUNCH_FAILED: i32 = 719;
    pub const NOT_SUPPORTED: i32 = 801;
    pub const UNKNOWN: i32 = 999;
}

/// A CUDA error code with a human-readable reason for the server log and the
/// client's diagnostic channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuError {
    pub code: i32,
    pub reason: String,
}

impl CuError {
    pub fn new(code: i32, reason: impl Into<String>) -> Self {
        Self {
            code,
            reason: reason.into(),
        }
    }
}

pub type CuResult<T> = Result<T, CuError>;
