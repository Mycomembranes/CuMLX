// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The backend seam and the host-only `loopback` test double.

use std::collections::{BTreeMap, HashMap};

use crate::{cu, CuError, CuResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    /// `None` when the backend cannot report it.
    pub total_mem: Option<u64>,
    pub sm_major: u32,
    pub sm_minor: u32,
}

/// One launch of a recorded graph: what `Backend::launch` takes, kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphLaunch {
    pub function: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem_bytes: u32,
    pub params: Vec<Vec<u8>>,
}

/// Element types a typed allocation may name (the served engine's set).
pub const DTYPES: [(&str, u64); 4] = [("float32", 4), ("float16", 2), ("int32", 4), ("uint8", 1)];

/// Size in bytes of one element of `dtype`, or an error for an unknown name or
/// a byte count that is not a whole number of elements.
pub fn check_dtype(dtype: &str, size: u64) -> CuResult<u64> {
    let item = DTYPES
        .iter()
        .find(|(name, _)| *name == dtype)
        .map(|(_, item)| *item)
        .ok_or_else(|| {
            CuError::new(cu::INVALID_VALUE, format!("unknown element type `{dtype}`"))
        })?;
    if !size.is_multiple_of(item) {
        return Err(CuError::new(
            cu::INVALID_VALUE,
            format!("{size} bytes is not a whole number of {dtype} elements"),
        ));
    }
    Ok(item)
}

/// What the adapter needs from the thing that actually owns the device.
///
/// Device pointers are plain addresses: `base + offset` must be accepted
/// wherever a pointer is taken. Launch parameters arrive as one byte string
/// per kernel parameter, already sized from the module's ABI.
pub trait Backend: Send {
    fn name(&self) -> &'static str;
    /// Optional capabilities, reported to clients in `hello`: `typed-alloc`,
    /// `graphs`. Valid after `init`.
    fn features(&self) -> Vec<&'static str>;
    fn init(&mut self) -> CuResult<()>;
    /// Exact identity of the served stack behind this backend, as a map with
    /// `pci_device`, `hostBootSessionID`, `sessionKey`, `stackID` and
    /// `inventoryDigest`. Qualification receipts are bound to it.
    fn identity(&mut self) -> CuResult<rmpv::Value>;
    fn driver_version(&mut self) -> CuResult<i32>;
    fn device_count(&mut self) -> CuResult<u32>;
    fn device_info(&mut self, device: u32) -> CuResult<DeviceInfo>;
    fn device_attribute(&mut self, device: u32, attribute: i32) -> CuResult<i32>;
    fn ctx_create(&mut self, device: u32, flags: u32) -> CuResult<u64>;
    fn ctx_destroy(&mut self, ctx: u64) -> CuResult<()>;
    fn synchronize(&mut self) -> CuResult<()>;
    /// `dtype` names the element type for owners that check it (the served
    /// engine's launch validators do); `None` is untyped CUDA memory.
    fn mem_alloc(&mut self, size: u64, dtype: Option<&str>) -> CuResult<u64>;
    fn mem_free(&mut self, dptr: u64) -> CuResult<()>;
    /// `(free, total)` bytes.
    fn mem_info(&mut self) -> CuResult<(u64, u64)>;
    fn copy_h2d(&mut self, dptr: u64, data: &[u8]) -> CuResult<()>;
    fn copy_d2h(&mut self, dptr: u64, len: usize) -> CuResult<Vec<u8>>;
    /// Copy `len` bytes from host memory at `src` to the device.
    ///
    /// # Safety
    /// `src` must be valid for reads of `len` bytes for the whole call. The
    /// memory may be written concurrently by another process (a shared
    /// segment), so implementations must not form a Rust reference to it.
    unsafe fn copy_h2d_raw(&mut self, dptr: u64, src: *const u8, len: usize) -> CuResult<()>;
    /// Copy `len` bytes from the device to host memory at `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for writes of `len` bytes for the whole call; the
    /// same aliasing rule as [`Backend::copy_h2d_raw`] applies.
    unsafe fn copy_d2h_raw(&mut self, dptr: u64, dst: *mut u8, len: usize) -> CuResult<()>;
    fn module_load(&mut self, cubin: &[u8]) -> CuResult<u64>;
    fn module_unload(&mut self, module: u64) -> CuResult<()>;
    fn function_get(&mut self, module: u64, name: &str) -> CuResult<u64>;
    fn launch(
        &mut self,
        function: u64,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        params: &[Vec<u8>],
    ) -> CuResult<()>;
    /// Record an ordered launch list once. Returns a graph handle.
    fn graph_create(&mut self, launches: &[GraphLaunch]) -> CuResult<u64>;
    /// Replay a graph `count` times, each replay complete before the next.
    fn graph_launch(&mut self, graph: u64, count: u32) -> CuResult<()>;
    fn graph_destroy(&mut self, graph: u64) -> CuResult<()>;
}

/// Host-only test double. Buffers live in host memory. It executes two
/// reference kernels, `vecadd` (`c[i] = a[i] + b[i]` over f32 with parameters
/// `(a, b, c, n: u32)`) and `matmul_macos` (see [`Loopback::launch_matmul`]),
/// so the wire, the forwarder, the shared-memory path and the card-backed
/// cuBLASLt layer can be tested end to end with no card. Every other launch is
/// refused.
pub struct Loopback {
    initialized: bool,
    next_va: u64,
    next_handle: u64,
    buffers: BTreeMap<u64, Vec<u8>>,
    contexts: Vec<u64>,
    modules: Vec<u64>,
    functions: HashMap<u64, String>,
    graphs: HashMap<u64, Vec<GraphLaunch>>,
}

const LOOPBACK_TOTAL_MEM: u64 = 1 << 30;
/// First device address. Kept low so the legacy 32-bit `CUdeviceptr` entry
/// points of the CUDA 3.x ABI can still name small allocations.
const LOOPBACK_VA_BASE: u64 = 0x0100_0000;
const VA_ALIGN: u64 = 0x1_0000;

impl Default for Loopback {
    fn default() -> Self {
        Self {
            initialized: false,
            next_va: LOOPBACK_VA_BASE,
            next_handle: 0x10,
            buffers: BTreeMap::new(),
            contexts: Vec::new(),
            modules: Vec::new(),
            functions: HashMap::new(),
            graphs: HashMap::new(),
        }
    }
}

impl Loopback {
    fn ready(&self) -> CuResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(CuError::new(
                cu::NOT_INITIALIZED,
                "cuInit has not been called",
            ))
        }
    }

    fn handle(&mut self) -> u64 {
        self.next_handle += 0x10;
        self.next_handle
    }

    fn used(&self) -> u64 {
        self.buffers.values().map(|b| b.len() as u64).sum()
    }

    /// Resolve `[dptr, dptr + len)` to its buffer base and offset.
    fn resolve(&self, dptr: u64, len: usize) -> CuResult<(u64, usize)> {
        let (base, buffer) = self.buffers.range(..=dptr).next_back().ok_or_else(|| {
            CuError::new(
                cu::INVALID_VALUE,
                format!("0x{dptr:x} is not a device pointer"),
            )
        })?;
        let offset = (dptr - base) as usize;
        if offset.checked_add(len).is_none_or(|end| end > buffer.len()) {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!(
                    "range 0x{dptr:x}+{len} leaves its {}-byte allocation",
                    buffer.len()
                ),
            ));
        }
        Ok((*base, offset))
    }

    fn read(&self, dptr: u64, len: usize) -> CuResult<Vec<u8>> {
        let (base, offset) = self.resolve(dptr, len)?;
        Ok(self.buffers[&base][offset..offset + len].to_vec())
    }

    fn write(&mut self, dptr: u64, data: &[u8]) -> CuResult<()> {
        let (base, offset) = self.resolve(dptr, data.len())?;
        self.buffers.get_mut(&base).expect("resolved base")[offset..offset + data.len()]
            .copy_from_slice(data);
        Ok(())
    }
}

fn param_u32(params: &[Vec<u8>], index: usize) -> CuResult<u32> {
    let bytes: [u8; 4] = params
        .get(index)
        .and_then(|p| p.as_slice().try_into().ok())
        .ok_or_else(|| {
            CuError::new(
                cu::INVALID_VALUE,
                format!("parameter {index} is not 4 bytes"),
            )
        })?;
    Ok(u32::from_le_bytes(bytes))
}

impl Loopback {
    /// Reference `matmul_macos`: `C[M,N] = A[M,K] @ B[K,N]`, row-major f32,
    /// parameters `(a, b, c, m, n, k)`, over the rows and columns the grid of
    /// 16x16 tiles covers. Accumulates in f64 and rounds to f32, like the
    /// resident engine's CPU model of the same kernel.
    fn launch_matmul(
        &mut self,
        grid: [u32; 3],
        block: [u32; 3],
        params: &[Vec<u8>],
    ) -> CuResult<()> {
        if block != [16, 16, 1] {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!("matmul_macos needs block (16,16,1), got {block:?}"),
            ));
        }
        let (a, b, c) = (
            param_u64(params, 0)?,
            param_u64(params, 1)?,
            param_u64(params, 2)?,
        );
        let (m, n, k) = (
            param_u32(params, 3)? as usize,
            param_u32(params, 4)? as usize,
            param_u32(params, 5)? as usize,
        );
        let too_big = || CuError::new(cu::INVALID_VALUE, "matmul dimensions overflow");
        let lhs = self.read(
            a,
            m.checked_mul(k)
                .and_then(|e| e.checked_mul(4))
                .ok_or_else(too_big)?,
        )?;
        let rhs = self.read(
            b,
            k.checked_mul(n)
                .and_then(|e| e.checked_mul(4))
                .ok_or_else(too_big)?,
        )?;
        let mut out = self.read(
            c,
            m.checked_mul(n)
                .and_then(|e| e.checked_mul(4))
                .ok_or_else(too_big)?,
        )?;
        let at = |bytes: &[u8], index: usize| {
            f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().expect("4 bytes")) as f64
        };
        let rows = m.min(grid[1] as usize * 16);
        let cols = n.min(grid[0] as usize * 16);
        for row in 0..rows {
            for col in 0..cols {
                let sum: f64 = (0..k)
                    .map(|i| at(&lhs, row * k + i) * at(&rhs, i * n + col))
                    .sum();
                out[(row * n + col) * 4..(row * n + col) * 4 + 4]
                    .copy_from_slice(&(sum as f32).to_le_bytes());
            }
        }
        self.write(c, &out)
    }
}

fn param_u64(params: &[Vec<u8>], index: usize) -> CuResult<u64> {
    let bytes: [u8; 8] = params
        .get(index)
        .and_then(|p| p.as_slice().try_into().ok())
        .ok_or_else(|| {
            CuError::new(
                cu::INVALID_VALUE,
                format!("parameter {index} is not 8 bytes"),
            )
        })?;
    Ok(u64::from_le_bytes(bytes))
}

impl Backend for Loopback {
    fn name(&self) -> &'static str {
        "loopback"
    }

    fn features(&self) -> Vec<&'static str> {
        vec!["typed-alloc", "graphs"]
    }

    fn init(&mut self) -> CuResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn identity(&mut self) -> CuResult<rmpv::Value> {
        Err(CuError::new(
            cu::NOT_SUPPORTED,
            "the loopback backend is host memory: there is no served stack to identify",
        ))
    }

    fn driver_version(&mut self) -> CuResult<i32> {
        Ok(10010)
    }

    fn device_count(&mut self) -> CuResult<u32> {
        self.ready()?;
        Ok(1)
    }

    fn device_info(&mut self, device: u32) -> CuResult<DeviceInfo> {
        self.ready()?;
        if device != 0 {
            return Err(CuError::new(
                cu::INVALID_DEVICE,
                format!("no device {device}"),
            ));
        }
        Ok(DeviceInfo {
            name: "cuda-adapter loopback (host memory, no GPU)".to_string(),
            total_mem: Some(LOOPBACK_TOTAL_MEM),
            sm_major: 12,
            sm_minor: 0,
        })
    }

    fn device_attribute(&mut self, device: u32, attribute: i32) -> CuResult<i32> {
        self.device_info(device)?;
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR.
        match attribute {
            75 => Ok(12),
            76 => Ok(0),
            _ => Err(CuError::new(
                cu::NOT_SUPPORTED,
                format!("device attribute {attribute} is not modelled"),
            )),
        }
    }

    fn ctx_create(&mut self, device: u32, _flags: u32) -> CuResult<u64> {
        self.device_info(device)?;
        let ctx = self.handle();
        self.contexts.push(ctx);
        Ok(ctx)
    }

    fn ctx_destroy(&mut self, ctx: u64) -> CuResult<()> {
        let before = self.contexts.len();
        self.contexts.retain(|c| *c != ctx);
        if self.contexts.len() == before {
            return Err(CuError::new(
                cu::INVALID_CONTEXT,
                format!("unknown context 0x{ctx:x}"),
            ));
        }
        Ok(())
    }

    fn synchronize(&mut self) -> CuResult<()> {
        self.ready()
    }

    fn mem_alloc(&mut self, size: u64, dtype: Option<&str>) -> CuResult<u64> {
        self.ready()?;
        if size == 0 {
            return Err(CuError::new(cu::INVALID_VALUE, "zero-byte allocation"));
        }
        if let Some(dtype) = dtype {
            check_dtype(dtype, size)?; // host memory has no element type; the name is only checked
        }
        if self.used().saturating_add(size) > LOOPBACK_TOTAL_MEM {
            return Err(CuError::new(
                cu::OUT_OF_MEMORY,
                format!("{size} bytes exceed the loopback pool"),
            ));
        }
        let base = self.next_va;
        self.next_va = (base + size).div_ceil(VA_ALIGN) * VA_ALIGN + VA_ALIGN;
        self.buffers.insert(base, vec![0u8; size as usize]);
        Ok(base)
    }

    fn mem_free(&mut self, dptr: u64) -> CuResult<()> {
        self.buffers.remove(&dptr).map(|_| ()).ok_or_else(|| {
            CuError::new(
                cu::INVALID_VALUE,
                format!("0x{dptr:x} is not an allocation base"),
            )
        })
    }

    fn mem_info(&mut self) -> CuResult<(u64, u64)> {
        self.ready()?;
        Ok((LOOPBACK_TOTAL_MEM - self.used(), LOOPBACK_TOTAL_MEM))
    }

    fn copy_h2d(&mut self, dptr: u64, data: &[u8]) -> CuResult<()> {
        self.write(dptr, data)
    }

    fn copy_d2h(&mut self, dptr: u64, len: usize) -> CuResult<Vec<u8>> {
        self.read(dptr, len)
    }

    unsafe fn copy_h2d_raw(&mut self, dptr: u64, src: *const u8, len: usize) -> CuResult<()> {
        let (base, offset) = self.resolve(dptr, len)?;
        let buffer = self.buffers.get_mut(&base).expect("resolved base");
        // SAFETY: the caller guarantees `src`; the destination range was resolved above.
        unsafe { std::ptr::copy_nonoverlapping(src, buffer.as_mut_ptr().add(offset), len) };
        Ok(())
    }

    unsafe fn copy_d2h_raw(&mut self, dptr: u64, dst: *mut u8, len: usize) -> CuResult<()> {
        let (base, offset) = self.resolve(dptr, len)?;
        // SAFETY: the caller guarantees `dst`; the source range was resolved above.
        unsafe {
            std::ptr::copy_nonoverlapping(self.buffers[&base].as_ptr().add(offset), dst, len)
        };
        Ok(())
    }

    fn module_load(&mut self, _cubin: &[u8]) -> CuResult<u64> {
        self.ready()?;
        let module = self.handle();
        self.modules.push(module);
        Ok(module)
    }

    fn module_unload(&mut self, module: u64) -> CuResult<()> {
        let before = self.modules.len();
        self.modules.retain(|m| *m != module);
        if self.modules.len() == before {
            return Err(CuError::new(
                cu::INVALID_HANDLE,
                format!("unknown module 0x{module:x}"),
            ));
        }
        Ok(())
    }

    fn function_get(&mut self, module: u64, name: &str) -> CuResult<u64> {
        if !self.modules.contains(&module) {
            return Err(CuError::new(
                cu::INVALID_HANDLE,
                format!("unknown module 0x{module:x}"),
            ));
        }
        let function = self.handle();
        self.functions.insert(function, name.to_string());
        Ok(function)
    }

    fn launch(
        &mut self,
        function: u64,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem_bytes: u32,
        params: &[Vec<u8>],
    ) -> CuResult<()> {
        let name = self
            .functions
            .get(&function)
            .ok_or_else(|| {
                CuError::new(
                    cu::INVALID_HANDLE,
                    format!("unknown function 0x{function:x}"),
                )
            })?
            .clone();
        if name == "matmul_macos" {
            return self.launch_matmul(grid, block, params);
        }
        if !name.to_ascii_lowercase().contains("vecadd") {
            return Err(CuError::new(
                cu::NOT_SUPPORTED,
                format!("the loopback backend executes only its vecadd and matmul_macos reference kernels, not `{name}`"),
            ));
        }
        let (a, b, c) = (
            param_u64(params, 0)?,
            param_u64(params, 1)?,
            param_u64(params, 2)?,
        );
        let n: [u8; 4] = params
            .get(3)
            .and_then(|p| p.as_slice().try_into().ok())
            .ok_or_else(|| {
                CuError::new(cu::INVALID_VALUE, "vecadd parameter 3 (n) is not 4 bytes")
            })?;
        let threads = grid
            .iter()
            .chain(block.iter())
            .map(|d| *d as u64)
            .product::<u64>();
        let count = (u32::from_le_bytes(n) as u64).min(threads) as usize;
        let lhs = self.read(a, count * 4)?;
        let rhs = self.read(b, count * 4)?;
        let mut out = Vec::with_capacity(count * 4);
        for (x, y) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
            let sum = f32::from_le_bytes(x.try_into().expect("4 bytes"))
                + f32::from_le_bytes(y.try_into().expect("4 bytes"));
            out.extend_from_slice(&sum.to_le_bytes());
        }
        self.write(c, &out)
    }

    fn graph_create(&mut self, launches: &[GraphLaunch]) -> CuResult<u64> {
        self.ready()?;
        if launches.is_empty() {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                "a graph needs at least one launch",
            ));
        }
        for launch in launches {
            if !self.functions.contains_key(&launch.function) {
                return Err(CuError::new(
                    cu::INVALID_HANDLE,
                    format!("unknown function 0x{:x}", launch.function),
                ));
            }
        }
        let graph = self.handle();
        self.graphs.insert(graph, launches.to_vec());
        Ok(graph)
    }

    fn graph_launch(&mut self, graph: u64, count: u32) -> CuResult<()> {
        let launches = self
            .graphs
            .get(&graph)
            .ok_or_else(|| CuError::new(cu::INVALID_HANDLE, format!("unknown graph 0x{graph:x}")))?
            .clone();
        for _ in 0..count {
            for launch in &launches {
                self.launch(
                    launch.function,
                    launch.grid,
                    launch.block,
                    launch.shared_mem_bytes,
                    &launch.params,
                )?;
            }
        }
        Ok(())
    }

    fn graph_destroy(&mut self, graph: u64) -> CuResult<()> {
        self.graphs
            .remove(&graph)
            .map(|_| ())
            .ok_or_else(|| CuError::new(cu::INVALID_HANDLE, format!("unknown graph 0x{graph:x}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_pointers_resolve_and_overruns_are_refused() {
        let mut lb = Loopback::default();
        lb.init().unwrap();
        let base = lb.mem_alloc(64, None).unwrap();
        lb.copy_h2d(base + 60, &[1, 2, 3, 4]).unwrap();
        assert_eq!(lb.copy_d2h(base + 60, 4).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(
            lb.copy_h2d(base + 61, &[0; 4]).unwrap_err().code,
            cu::INVALID_VALUE
        );
        assert_eq!(lb.mem_free(base + 4).unwrap_err().code, cu::INVALID_VALUE);
        lb.mem_free(base).unwrap();
        assert_eq!(lb.copy_d2h(base, 1).unwrap_err().code, cu::INVALID_VALUE);
    }

    #[test]
    fn calls_before_init_are_refused() {
        let mut lb = Loopback::default();
        assert_eq!(lb.mem_alloc(4, None).unwrap_err().code, cu::NOT_INITIALIZED);
        assert_eq!(lb.device_count().unwrap_err().code, cu::NOT_INITIALIZED);
    }

    #[test]
    fn matmul_reference_multiplies_row_major_and_respects_the_grid() {
        let mut lb = Loopback::default();
        lb.init().unwrap();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        // A is 2x3, B is 3x2, row-major.
        let (a, b, c) = (
            lb.mem_alloc(24, None).unwrap(),
            lb.mem_alloc(24, None).unwrap(),
            lb.mem_alloc(16, None).unwrap(),
        );
        lb.copy_h2d(a, &bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .unwrap();
        lb.copy_h2d(b, &bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
            .unwrap();
        let module = lb.module_load(&[]).unwrap();
        let f = lb.function_get(module, "matmul_macos").unwrap();
        let params = vec![
            a.to_le_bytes().to_vec(),
            b.to_le_bytes().to_vec(),
            c.to_le_bytes().to_vec(),
            2u32.to_le_bytes().to_vec(),
            2u32.to_le_bytes().to_vec(),
            3u32.to_le_bytes().to_vec(),
        ];
        lb.launch(f, [1, 1, 1], [16, 16, 1], 0, &params).unwrap();
        assert_eq!(
            lb.copy_d2h(c, 16).unwrap(),
            bytes(&[58.0, 64.0, 139.0, 154.0])
        );
        assert_eq!(
            lb.launch(f, [1, 1, 1], [8, 8, 1], 0, &params)
                .unwrap_err()
                .code,
            cu::INVALID_VALUE
        );
    }

    #[test]
    fn graph_replays_its_launch_list_and_typed_alloc_checks_the_size() {
        let mut lb = Loopback::default();
        lb.init().unwrap();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        assert_eq!(
            lb.mem_alloc(6, Some("float32")).unwrap_err().code,
            cu::INVALID_VALUE
        );
        assert_eq!(
            lb.mem_alloc(8, Some("float64")).unwrap_err().code,
            cu::INVALID_VALUE
        );
        let a = lb.mem_alloc(8, Some("float32")).unwrap();
        let (b, c) = (
            lb.mem_alloc(8, None).unwrap(),
            lb.mem_alloc(8, None).unwrap(),
        );
        lb.copy_h2d(a, &bytes(&[1.0, 2.0])).unwrap();
        lb.copy_h2d(b, &bytes(&[10.0, 10.0])).unwrap();
        let module = lb.module_load(&[]).unwrap();
        let f = lb.function_get(module, "vecadd_macos").unwrap();
        let p = |x: u64, y: u64, z: u64| {
            vec![
                x.to_le_bytes().to_vec(),
                y.to_le_bytes().to_vec(),
                z.to_le_bytes().to_vec(),
                2u32.to_le_bytes().to_vec(),
            ]
        };
        let step = |params| GraphLaunch {
            function: f,
            grid: [1, 1, 1],
            block: [256, 1, 1],
            shared_mem_bytes: 0,
            params,
        };
        // c = a + b; then a = c + b: each replay adds 20 to a.
        let graph = lb
            .graph_create(&[step(p(a, b, c)), step(p(c, b, a))])
            .unwrap();
        lb.graph_launch(graph, 3).unwrap();
        assert_eq!(lb.copy_d2h(a, 8).unwrap(), bytes(&[61.0, 62.0]));
        assert_eq!(lb.graph_create(&[]).unwrap_err().code, cu::INVALID_VALUE);
        lb.graph_destroy(graph).unwrap();
        assert_eq!(
            lb.graph_launch(graph, 1).unwrap_err().code,
            cu::INVALID_HANDLE
        );
    }

    #[test]
    fn vecadd_runs_and_other_kernels_are_refused() {
        let mut lb = Loopback::default();
        lb.init().unwrap();
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (a, b, c) = (
            lb.mem_alloc(16, None).unwrap(),
            lb.mem_alloc(16, None).unwrap(),
            lb.mem_alloc(16, None).unwrap(),
        );
        lb.copy_h2d(a, &bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        lb.copy_h2d(b, &bytes(&[0.5, 0.5, 0.5, 0.5])).unwrap();
        let module = lb.module_load(&[]).unwrap();
        let params = vec![
            a.to_le_bytes().to_vec(),
            b.to_le_bytes().to_vec(),
            c.to_le_bytes().to_vec(),
            3u32.to_le_bytes().to_vec(),
        ];
        let f = lb.function_get(module, "vecadd_macos").unwrap();
        lb.launch(f, [1, 1, 1], [256, 1, 1], 0, &params).unwrap();
        // n = 3: the fourth element stays zero.
        assert_eq!(lb.copy_d2h(c, 16).unwrap(), bytes(&[1.5, 2.5, 3.5, 0.0]));
        let other = lb.function_get(module, "rms_norm").unwrap();
        assert_eq!(
            lb.launch(other, [1, 1, 1], [1, 1, 1], 0, &params)
                .unwrap_err()
                .code,
            cu::NOT_SUPPORTED
        );
    }
}
