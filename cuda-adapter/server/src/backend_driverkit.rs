// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend that reaches a served card through the `cuda-driverkit` C ABI.
//!
//! `cuInit` (with `CUDA_OXIDE_CUDA_BACKEND=pciserver`) finds the exact-serve
//! launch endpoint, checks its identity and authenticates. Everything after
//! that is a request to that endpoint's shared CUDA driver, sent through
//! `c_abi::shared_cuda_request` on the same authenticated connection. This
//! process never opens DriverKit; the card keeps one owner (`pci_server_full`).
//!
//! Device addresses handed to clients are this backend's own: one range per
//! allocation. The shared driver knows allocations by handle, so a client
//! address is translated to `(handle, offset)` wherever one is taken, and an
//! eight-byte kernel parameter whose value lies inside a live range is passed
//! as a device pointer.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use cuda_driverkit::c_abi as abi;
use rmpv::Value;

use crate::backend::{Backend, DeviceInfo, GraphLaunch, check_dtype};
use crate::{CuError, CuResult, cu};

/// First client-visible device address. High, so a scalar parameter is not
/// mistaken for a pointer.
const VA_BASE: u64 = 0x7ad0_0000_0000;
const VA_ALIGN: u64 = 0x1_0000;
/// Bytes per copy request (the served engine caps one transfer at 48 MiB).
const COPY_CHUNK: usize = 16 * 1024 * 1024;

struct Allocation {
    handle: u64,
    size: u64,
}

pub struct DriverKit {
    next_va: u64,
    /// client base address -> the shared driver's handle and the size
    allocations: BTreeMap<u64, Allocation>,
    /// Feature names from the shared driver's `rpc_describe`.
    server_features: HashSet<String>,
    /// Bytes one copy request carries: `COPY_CHUNK`, or less when the owner says so.
    copy_chunk: usize,
    /// That reply, kept for `identity`.
    described: Value,
}

impl Default for DriverKit {
    fn default() -> Self {
        Self {
            next_va: VA_BASE,
            allocations: BTreeMap::new(),
            server_features: HashSet::new(),
            copy_chunk: COPY_CHUNK,
            described: Value::Nil,
        }
    }
}

fn check(code: i32, what: &str) -> CuResult<()> {
    if code == cu::SUCCESS {
        Ok(())
    } else {
        Err(CuError::new(
            code,
            format!("{what} returned CUDA error {code}"),
        ))
    }
}

fn pair(key: &str, value: Value) -> (Value, Value) {
    (Value::from(key), value)
}

fn field<'a>(map: &'a Value, key: &str) -> Option<&'a Value> {
    map.as_map()?
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

fn field_u64(map: &Value, key: &str, what: &str) -> CuResult<u64> {
    field(map, key).and_then(Value::as_u64).ok_or_else(|| {
        CuError::new(
            cu::UNKNOWN,
            format!("{what}: the shared driver's reply has no `{key}`"),
        )
    })
}

/// Set when a request to the shared driver ended without its own reply: a
/// read timeout, a closed socket, or a reply with another sequence number.
/// The reply may still arrive later and would be read as the answer to the
/// next request, so nothing more is sent on this connection, cleanup included.
/// (2026-11-20: a download timed out here while the card stalled, and the
/// cleanup that followed read each earlier reply as its own.)
static CONNECTION_LOST: AtomicBool = AtomicBool::new(false);

fn transport_failure(reason: &str) -> bool {
    reason.contains("io:")
        || reason.contains("seq mismatch")
        || reason.contains("wire encode/decode")
}

/// Send one request. A refusal (the driver validated and said no) is
/// `INVALID_VALUE`; a device fault or an unhealthy owner is `UNKNOWN`.
fn request(op: &str, args: Vec<(Value, Value)>) -> CuResult<Value> {
    if CONNECTION_LOST.load(Ordering::SeqCst) {
        return Err(CuError::new(
            cu::UNKNOWN,
            format!(
                "{op}: not sent: an earlier request to the shared driver lost its reply \
                 (timeout or a reply out of sequence), so the connection cannot be trusted"
            ),
        ));
    }
    abi::shared_cuda_request(op, Value::Map(args)).map_err(|reason| {
        if transport_failure(&reason) {
            CONNECTION_LOST.store(true, Ordering::SeqCst);
        }
        let code = if reason.to_ascii_lowercase().contains("device fault")
            || reason.contains("unhealthy")
            || reason.contains("transport")
        {
            cu::UNKNOWN
        } else if reason.contains("unsupported shared CUDA operation") {
            cu::NOT_SUPPORTED
        } else {
            cu::INVALID_VALUE
        };
        CuError::new(code, format!("{op}: {reason}"))
    })
}

impl DriverKit {
    fn has(&self, feature: &str) -> bool {
        self.server_features.contains(feature)
    }

    /// Split a device address into the allocation that contains
    /// `[dptr, dptr + len)` and the byte offset inside it.
    fn locate(&self, dptr: u64, len: u64) -> CuResult<(&Allocation, u64)> {
        let (base, allocation) = self.allocations.range(..=dptr).next_back().ok_or_else(|| {
            CuError::new(
                cu::INVALID_VALUE,
                format!("0x{dptr:x} is not a device pointer"),
            )
        })?;
        let offset = dptr - base;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > allocation.size)
        {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!(
                    "range 0x{dptr:x}+{len} leaves its {}-byte allocation",
                    allocation.size
                ),
            ));
        }
        Ok((allocation, offset))
    }

    /// The allocation an address points into, if any (for kernel parameters).
    fn pointer(&self, value: u64) -> Option<(u64, u64)> {
        let (base, allocation) = self.allocations.range(..=value).next_back()?;
        (value - base < allocation.size).then_some((allocation.handle, value - base))
    }

    fn upload(&self, handle: u64, offset: u64, data: &[u8]) -> CuResult<()> {
        let mut args = vec![
            pair("dst", Value::from(handle)),
            pair("data", Value::Binary(data.to_vec())),
        ];
        if offset != 0 {
            args.push(pair("offset", Value::from(offset)));
        }
        request("copy_h2d", args).map(|_| ())
    }

    fn download(&self, handle: u64, offset: u64, len: usize) -> CuResult<Vec<u8>> {
        let mut args = vec![
            pair("src", Value::from(handle)),
            pair("size", Value::from(len as u64)),
        ];
        if offset != 0 {
            args.push(pair("offset", Value::from(offset)));
        }
        let reply = request("copy_d2h", args)?;
        match field(&reply, "data") {
            Some(Value::Binary(bytes)) if bytes.len() == len => Ok(bytes.clone()),
            _ => Err(CuError::new(
                cu::UNKNOWN,
                "copy_d2h: the shared driver returned the wrong amount of data",
            )),
        }
    }

    /// The reflected form of one launch: integer parameter bits, which of them
    /// are device pointers (then the value is the handle), and pointer offsets.
    fn launch_map(&self, launch: &GraphLaunch) -> CuResult<Vec<(Value, Value)>> {
        let mut values = Vec::with_capacity(launch.params.len());
        let mut pointers = Vec::with_capacity(launch.params.len());
        let mut offsets = Vec::with_capacity(launch.params.len());
        for (index, bytes) in launch.params.iter().enumerate() {
            if !matches!(bytes.len(), 1 | 2 | 4 | 8) {
                return Err(CuError::new(
                    cu::NOT_SUPPORTED,
                    format!(
                        "parameter {index} is {} bytes; the served stack takes 1, 2, 4 or 8",
                        bytes.len()
                    ),
                ));
            }
            let mut word = [0u8; 8];
            word[..bytes.len()].copy_from_slice(bytes);
            let value = u64::from_le_bytes(word);
            match (bytes.len() == 8).then(|| self.pointer(value)).flatten() {
                Some((handle, offset)) => {
                    values.push(Value::from(handle));
                    pointers.push(Value::from(true));
                    offsets.push(Value::from(offset));
                }
                None => {
                    values.push(Value::from(value));
                    pointers.push(Value::from(false));
                    offsets.push(Value::from(0));
                }
            }
        }
        let dims = |d: [u32; 3]| Value::Array(d.iter().map(|v| Value::from(*v)).collect());
        Ok(vec![
            pair("func", Value::from(launch.function)),
            pair("grid", dims(launch.grid)),
            pair("block", dims(launch.block)),
            pair("smem", Value::from(launch.shared_mem_bytes)),
            pair("stream", Value::from(0)),
            pair("params", Value::Array(values)),
            pair("parameter_pointers", Value::Array(pointers)),
            pair("parameter_offsets", Value::Array(offsets)),
        ])
    }
}

impl Backend for DriverKit {
    fn name(&self) -> &'static str {
        "driverkit"
    }

    fn features(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.has("typed-alloc") {
            out.push("typed-alloc");
        }
        if self.has("pinned-graphs") {
            out.push("graphs");
        }
        out
    }

    fn init(&mut self) -> CuResult<()> {
        // SAFETY: cuInit takes no pointers.
        check(unsafe { abi::cuInit(0) }, "cuInit")?;
        // What this owner offers. An older owner has no `features` beyond the
        // base set: typed allocation and graphs are then refused here, and
        // interior copies fall back to whole-prefix copies.
        let described = request("rpc_describe", Vec::new())?;
        self.server_features = field(&described, "features")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        // An owner may cap one copy below this adapter's own chunk (`max_copy_bytes`).
        // Whole words; a cap that leaves no room for one is not an owner to copy through.
        self.copy_chunk = match field(&described, "max_copy_bytes").and_then(Value::as_u64) {
            None => COPY_CHUNK,
            Some(cap) if cap >= 4 => (cap.min(COPY_CHUNK as u64) as usize) & !3,
            Some(cap) => {
                return Err(CuError::new(
                    cu::UNKNOWN,
                    format!("the shared driver's max_copy_bytes is {cap}"),
                ));
            }
        };
        self.described = described;
        Ok(())
    }

    fn identity(&mut self) -> CuResult<Value> {
        // The launch endpoint's own identity (which the C ABI checked against the
        // serve registry before it authenticated), under the names the tensor
        // server uses.
        let missing = |key: &str| {
            CuError::new(
                cu::UNKNOWN,
                format!("the shared driver's identity has no `{key}`"),
            )
        };
        let device = field(&self.described, "pci_device")
            .and_then(Value::as_u64)
            .ok_or_else(|| missing("pci_device"))?;
        let mut out = vec![pair("pci_device", Value::from(device))];
        for (name, wire) in [
            ("hostBootSessionID", "host_boot_session_id"),
            ("sessionKey", "session_key"),
            ("stackID", "stack_id"),
            ("inventoryDigest", "inventory_digest"),
        ] {
            let value = field(&self.described, wire)
                .and_then(Value::as_str)
                .ok_or_else(|| missing(wire))?;
            out.push(pair(name, Value::from(value)));
        }
        Ok(Value::Map(out))
    }

    fn driver_version(&mut self) -> CuResult<i32> {
        let mut version = 0i32;
        // SAFETY: `version` is a valid, aligned out-pointer.
        check(
            unsafe { abi::cuDriverGetVersion(&mut version) },
            "cuDriverGetVersion",
        )?;
        Ok(version)
    }

    fn device_count(&mut self) -> CuResult<u32> {
        let mut count = 0i32;
        // SAFETY: `count` is a valid out-pointer.
        check(
            unsafe { abi::cuDeviceGetCount(&mut count) },
            "cuDeviceGetCount",
        )?;
        Ok(count.max(0) as u32)
    }

    fn device_info(&mut self, device: u32) -> CuResult<DeviceInfo> {
        let mut name = [0 as std::ffi::c_char; 128];
        // SAFETY: the buffer is 128 bytes and its length is passed.
        check(
            unsafe { abi::cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, device as i32) },
            "cuDeviceGetName",
        )?;
        // SAFETY: cuDeviceGetName NUL-terminates within the given length.
        let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok(DeviceInfo {
            name,
            total_mem: None,
            sm_major: 12,
            sm_minor: 0,
        })
    }

    fn device_attribute(&mut self, device: u32, attribute: i32) -> CuResult<i32> {
        let mut value = 0i32;
        // SAFETY: `value` is a valid out-pointer.
        check(
            unsafe { abi::cuDeviceGetAttribute(&mut value, attribute, device as i32) },
            "cuDeviceGetAttribute",
        )?;
        Ok(value)
    }

    fn ctx_create(&mut self, device: u32, flags: u32) -> CuResult<u64> {
        let mut ctx: abi::CUcontext = std::ptr::null_mut();
        // SAFETY: `ctx` is a valid, aligned out-pointer.
        check(
            unsafe { abi::cuCtxCreate(&mut ctx, flags, device as i32) },
            "cuCtxCreate",
        )?;
        Ok(ctx as u64)
    }

    fn ctx_destroy(&mut self, ctx: u64) -> CuResult<()> {
        // SAFETY: the handle is an opaque id owned by the C ABI's registry.
        check(
            unsafe { abi::cuCtxDestroy(ctx as abi::CUcontext) },
            "cuCtxDestroy",
        )
    }

    fn synchronize(&mut self) -> CuResult<()> {
        request("ctx_sync", Vec::new()).map(|_| ())
    }

    fn mem_alloc(&mut self, size: u64, dtype: Option<&str>) -> CuResult<u64> {
        if size == 0 {
            return Err(CuError::new(cu::INVALID_VALUE, "zero-byte allocation"));
        }
        let mut args = vec![pair("size", Value::from(size))];
        if let Some(dtype) = dtype {
            check_dtype(dtype, size)?;
            if !self.has("typed-alloc") {
                return Err(CuError::new(
                    cu::NOT_SUPPORTED,
                    "the served stack predates typed allocation; re-serve it with the current scripts",
                ));
            }
            args.push(pair("dtype", Value::from(dtype)));
        }
        let handle = field_u64(&request("alloc", args)?, "handle", "alloc")?;
        let base = self.next_va;
        self.next_va = (base + size).div_ceil(VA_ALIGN) * VA_ALIGN + VA_ALIGN;
        self.allocations.insert(base, Allocation { handle, size });
        Ok(base)
    }

    fn mem_free(&mut self, dptr: u64) -> CuResult<()> {
        let handle = self
            .allocations
            .get(&dptr)
            .ok_or_else(|| {
                CuError::new(
                    cu::INVALID_VALUE,
                    format!("0x{dptr:x} is not an allocation base"),
                )
            })?
            .handle;
        request("free", vec![pair("handle", Value::from(handle))])?;
        self.allocations.remove(&dptr);
        Ok(())
    }

    fn mem_info(&mut self) -> CuResult<(u64, u64)> {
        Err(CuError::new(
            cu::NOT_SUPPORTED,
            "the served stack does not report memory totals",
        ))
    }

    fn copy_h2d(&mut self, dptr: u64, data: &[u8]) -> CuResult<()> {
        let (allocation, offset) = self.locate(dptr, data.len() as u64)?;
        let handle = allocation.handle;
        if data.is_empty() {
            return Ok(());
        }
        if self.has("copy-offsets") {
            let mut at = 0usize;
            while at < data.len() {
                let end = (at + self.copy_chunk).min(data.len());
                self.upload(handle, offset + at as u64, &data[at..end])?;
                at = end;
            }
            return Ok(());
        }
        if offset == 0 {
            return self.upload(handle, 0, data);
        }
        // An owner without copy offsets copies whole prefixes only: read the
        // bytes before the write, append the new ones, write the prefix back.
        let mut prefix = self.download(handle, 0, offset as usize)?;
        prefix.extend_from_slice(data);
        self.upload(handle, 0, &prefix)
    }

    fn copy_d2h(&mut self, dptr: u64, len: usize) -> CuResult<Vec<u8>> {
        let (allocation, offset) = self.locate(dptr, len as u64)?;
        let handle = allocation.handle;
        if len == 0 {
            return Ok(Vec::new());
        }
        if self.has("copy-offsets") {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                let take = self.copy_chunk.min(len - out.len());
                out.extend_from_slice(&self.download(handle, offset + out.len() as u64, take)?);
            }
            return Ok(out);
        }
        if offset == 0 {
            return self.download(handle, 0, len);
        }
        let prefix = self.download(handle, 0, offset as usize + len)?;
        Ok(prefix[offset as usize..].to_vec())
    }

    unsafe fn copy_h2d_raw(&mut self, dptr: u64, src: *const u8, len: usize) -> CuResult<()> {
        // The request carries the bytes, so they are read out of the shared
        // segment once, here, through the raw pointer.
        let mut staged = vec![0u8; len];
        // SAFETY: the caller guarantees `src` for `len` bytes; `staged` holds `len`.
        unsafe { std::ptr::copy_nonoverlapping(src, staged.as_mut_ptr(), len) };
        self.copy_h2d(dptr, &staged)
    }

    unsafe fn copy_d2h_raw(&mut self, dptr: u64, dst: *mut u8, len: usize) -> CuResult<()> {
        let data = self.copy_d2h(dptr, len)?;
        // SAFETY: the caller guarantees `dst` for `len` bytes; `data` holds `len`.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, len) };
        Ok(())
    }

    fn module_load(&mut self, cubin: &[u8]) -> CuResult<u64> {
        let reply = request(
            "module_load",
            vec![pair("image", Value::Binary(cubin.to_vec()))],
        )?;
        field_u64(&reply, "module", "module_load")
    }

    fn module_unload(&mut self, module: u64) -> CuResult<()> {
        request("module_unload", vec![pair("module", Value::from(module))]).map(|_| ())
    }

    fn function_get(&mut self, module: u64, name: &str) -> CuResult<u64> {
        let reply = request(
            "function_get",
            vec![
                pair("module", Value::from(module)),
                pair("name", Value::from(name)),
            ],
        )?;
        field_u64(&reply, "function", "function_get")
    }

    fn launch(
        &mut self,
        function: u64,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        params: &[Vec<u8>],
    ) -> CuResult<()> {
        let launch = GraphLaunch {
            function,
            grid,
            block,
            shared_mem_bytes,
            params: params.to_vec(),
        };
        request("launch_kernel", self.launch_map(&launch)?).map(|_| ())
    }

    fn graph_create(&mut self, launches: &[GraphLaunch]) -> CuResult<u64> {
        if !self.has("pinned-graphs") {
            return Err(CuError::new(
                cu::NOT_SUPPORTED,
                "the served stack predates graphs; re-serve it with the current scripts",
            ));
        }
        let entries = launches
            .iter()
            .map(|launch| self.launch_map(launch).map(Value::Map))
            .collect::<CuResult<Vec<_>>>()?;
        let reply = request(
            "graph_create",
            vec![pair("launches", Value::Array(entries))],
        )?;
        field_u64(&reply, "graph", "graph_create")
    }

    fn graph_launch(&mut self, graph: u64, count: u32) -> CuResult<()> {
        request(
            "graph_launch",
            vec![
                pair("graph", Value::from(graph)),
                pair("count", Value::from(count)),
            ],
        )
        .map(|_| ())
    }

    fn graph_destroy(&mut self, graph: u64) -> CuResult<()> {
        request("graph_destroy", vec![pair("graph", Value::from(graph))]).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_allocation(base: u64, handle: u64, size: u64) -> DriverKit {
        let mut backend = DriverKit::default();
        backend
            .allocations
            .insert(base, Allocation { handle, size });
        backend
    }

    #[test]
    fn a_lost_reply_is_a_transport_failure_and_a_refusal_is_not() {
        // the reason seen on the card, 2026-11-20
        assert!(transport_failure(
            "copy_d2h: io: Resource temporarily unavailable (os error 35)"
        ));
        assert!(transport_failure(
            "free: wire encode/decode: response seq mismatch: expected 3728, got 3727"
        ));
        assert!(!transport_failure(
            "launch: grid exceeds the validator's bound"
        ));
        assert!(!transport_failure(
            "Device fault detected: shared CUDA download"
        ));
    }

    #[test]
    fn eight_byte_parameters_inside_an_allocation_become_handle_and_offset() {
        let backend = with_allocation(VA_BASE, 0x4001, 64);
        let launch = GraphLaunch {
            function: 7,
            grid: [2, 1, 1],
            block: [256, 1, 1],
            shared_mem_bytes: 0,
            params: vec![
                (VA_BASE + 16).to_le_bytes().to_vec(), // interior pointer
                (VA_BASE + 64).to_le_bytes().to_vec(), // one past the end: a plain number
                9u32.to_le_bytes().to_vec(),
            ],
        };
        let map = Value::Map(backend.launch_map(&launch).unwrap());
        let list = |key: &str| field(&map, key).and_then(Value::as_array).unwrap().clone();
        assert_eq!(
            list("params"),
            vec![
                Value::from(0x4001u64),
                Value::from(VA_BASE + 64),
                Value::from(9u64)
            ]
        );
        assert_eq!(
            list("parameter_pointers"),
            vec![Value::from(true), Value::from(false), Value::from(false)]
        );
        assert_eq!(
            list("parameter_offsets"),
            vec![Value::from(16u64), Value::from(0), Value::from(0)]
        );
        assert_eq!(field(&map, "func").and_then(Value::as_u64), Some(7));
    }

    #[test]
    fn odd_parameter_widths_are_refused_before_any_request() {
        let backend = DriverKit::default();
        let launch = GraphLaunch {
            function: 1,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem_bytes: 0,
            params: vec![vec![0u8; 12]],
        };
        assert_eq!(
            backend.launch_map(&launch).unwrap_err().code,
            cu::NOT_SUPPORTED
        );
    }

    #[test]
    fn ranges_are_located_and_overruns_refused() {
        let backend = with_allocation(VA_BASE, 5, 32);
        let (allocation, offset) = backend.locate(VA_BASE + 8, 24).unwrap();
        assert_eq!((allocation.handle, offset), (5, 8));
        assert_eq!(
            backend.locate(VA_BASE + 8, 25).err().unwrap().code,
            cu::INVALID_VALUE
        );
        assert_eq!(
            backend.locate(VA_BASE - 1, 1).err().unwrap().code,
            cu::INVALID_VALUE
        );
    }
}
