// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One client session: request dispatch, per-client resource tracking and
//! cleanup when the client goes away.

use std::collections::{BTreeSet, HashMap};
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rmpv::Value;

use crate::backend::{Backend, GraphLaunch};
use crate::module::{self, JitOptions, PtxasLocator};
use crate::proto::{self, arg_bin, arg_dims, arg_i32, arg_str, arg_u32, arg_u64, map};
use crate::shm::{FdChannel, Segment, MAX_SEGMENTS};
use crate::{cu, CuError, CuResult, PROTOCOL, SERVER_KIND};

/// Client ABI revision this server speaks.
pub const CLIENT_ABI: u64 = 1;
/// Most replays one `graph_launch` request may ask for.
pub const MAX_GRAPH_REPLAYS: u32 = 64;
/// Most launches one graph may hold.
pub const MAX_GRAPH_LAUNCHES: usize = 4096;
/// Largest single upload or download, matching the resident stack's cap.
pub const MAX_TRANSFER_BYTES: u64 = 768 * 1024 * 1024;

pub type SharedBackend = Arc<Mutex<Box<dyn Backend>>>;

/// State shared by every connection.
pub struct ServerState {
    pub backend: SharedBackend,
    pub ptxas: PtxasLocator,
    /// One client at a time: the card has one owner and one launch stream.
    busy: AtomicBool,
}

impl ServerState {
    pub fn new(backend: Box<dyn Backend>, ptxas: PtxasLocator) -> Arc<Self> {
        Arc::new(Self {
            backend: Arc::new(Mutex::new(backend)),
            ptxas,
            busy: AtomicBool::new(false),
        })
    }
}

struct LoadedModule {
    cubin: Vec<u8>,
}

struct Upload {
    dptr: u64,
    data: Vec<u8>,
    total: usize,
}

struct Download {
    dptr: u64,
    data: Vec<u8>,
}

/// Everything one client owns. Dropping the session returns it all.
pub struct Session {
    state: Arc<ServerState>,
    greeted: bool,
    contexts: BTreeSet<u64>,
    allocations: BTreeSet<u64>,
    modules: HashMap<u64, LoadedModule>,
    /// function handle -> (module, parameter sizes)
    functions: HashMap<u64, (u64, Vec<u32>)>,
    upload: Option<Upload>,
    download: Option<Download>,
    /// Graphs this client recorded.
    graphs: BTreeSet<u64>,
    /// Shared host segments, by handle.
    segments: HashMap<u64, Segment>,
    next_segment: u64,
    /// Descriptor to attach right after the reply to the current request.
    pending_fd: Option<RawFd>,
    stats: TransferStats,
}

/// Where a client's copy bytes travelled, reported when the client leaves.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TransferStats {
    /// Bytes carried as socket payload (ordinary host memory).
    pub socket_bytes: u64,
    /// Bytes copied straight from or to a shared host segment.
    pub shared_bytes: u64,
    pub launches: u64,
}

impl Session {
    pub fn new(state: Arc<ServerState>) -> Self {
        Self {
            state,
            greeted: false,
            contexts: BTreeSet::new(),
            allocations: BTreeSet::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
            upload: None,
            download: None,
            graphs: BTreeSet::new(),
            segments: HashMap::new(),
            next_segment: 0,
            pending_fd: None,
            stats: TransferStats::default(),
        }
    }

    fn backend(&self) -> CuResult<std::sync::MutexGuard<'_, Box<dyn Backend>>> {
        self.state
            .backend
            .lock()
            .map_err(|_| CuError::new(cu::UNKNOWN, "backend lock poisoned by an earlier panic"))
    }

    pub fn dispatch(&mut self, op: &str, args: &Value) -> CuResult<Value> {
        if op == "hello" {
            return self.hello(args);
        }
        if !self.greeted {
            return Err(CuError::new(
                cu::NOT_INITIALIZED,
                "first request must be `hello`",
            ));
        }
        match op {
            "init" => {
                self.backend()?.init()?;
                Ok(Value::Nil)
            }
            "driver_version" => Ok(Value::from(self.backend()?.driver_version()?)),
            "identity" => self.backend()?.identity(),
            "device_count" => Ok(Value::from(self.backend()?.device_count()?)),
            "device_info" => {
                let info = self.backend()?.device_info(arg_u32(args, "device")?)?;
                Ok(map(vec![
                    ("name", Value::from(info.name)),
                    (
                        "total_mem",
                        info.total_mem.map(Value::from).unwrap_or(Value::Nil),
                    ),
                    ("sm_major", Value::from(info.sm_major)),
                    ("sm_minor", Value::from(info.sm_minor)),
                ]))
            }
            "device_attr" => {
                let value = self
                    .backend()?
                    .device_attribute(arg_u32(args, "device")?, arg_i32(args, "attr")?)?;
                Ok(Value::from(value))
            }
            "ctx_create" => {
                let ctx = self
                    .backend()?
                    .ctx_create(arg_u32(args, "device")?, arg_u32(args, "flags")?)?;
                self.contexts.insert(ctx);
                Ok(Value::from(ctx))
            }
            "ctx_destroy" => {
                let ctx = arg_u64(args, "ctx")?;
                if !self.contexts.contains(&ctx) {
                    return Err(CuError::new(
                        cu::INVALID_CONTEXT,
                        format!("context 0x{ctx:x} is not owned by this client"),
                    ));
                }
                self.backend()?.ctx_destroy(ctx)?;
                self.contexts.remove(&ctx);
                Ok(Value::Nil)
            }
            "sync" => {
                self.backend()?.synchronize()?;
                Ok(Value::Nil)
            }
            "mem_alloc" => {
                let size = arg_u64(args, "size")?;
                let dtype = match proto::get(args, "dtype") {
                    None | Some(Value::Nil) => None,
                    Some(value) => Some(value.as_str().ok_or_else(|| {
                        CuError::new(cu::INVALID_VALUE, "`dtype` must be a string")
                    })?),
                };
                let dptr = self.backend()?.mem_alloc(size, dtype)?;
                self.allocations.insert(dptr);
                Ok(Value::from(dptr))
            }
            "mem_free" => {
                let dptr = arg_u64(args, "dptr")?;
                if !self.allocations.contains(&dptr) {
                    return Err(CuError::new(
                        cu::INVALID_VALUE,
                        format!("0x{dptr:x} is not an allocation of this client"),
                    ));
                }
                self.backend()?.mem_free(dptr)?;
                self.allocations.remove(&dptr);
                Ok(Value::Nil)
            }
            "mem_info" => {
                let (free, total) = self.backend()?.mem_info()?;
                Ok(map(vec![
                    ("free", Value::from(free)),
                    ("total", Value::from(total)),
                ]))
            }
            "h2d" => self.h2d(args),
            "d2h" => self.d2h(args),
            "host_alloc" => self.host_alloc(args),
            "host_free" => {
                let handle = arg_u64(args, "handle")?;
                // Dropping the segment unmaps it here; the client unmaps its own view.
                self.segments
                    .remove(&handle)
                    .map(|_| Value::Nil)
                    .ok_or_else(|| {
                        CuError::new(cu::INVALID_VALUE, format!("unknown host segment {handle}"))
                    })
            }
            "h2d_shm" => {
                let (dptr, src, len) = self.segment_range(args)?;
                // SAFETY: `src` was bounds-checked against a live segment owned by
                // this session, which stays mapped until the call returns.
                unsafe { self.backend()?.copy_h2d_raw(dptr, src, len)? };
                self.stats.shared_bytes += len as u64;
                Ok(Value::Nil)
            }
            "d2h_shm" => {
                let (dptr, dst, len) = self.segment_range(args)?;
                // SAFETY: as for `h2d_shm`.
                unsafe { self.backend()?.copy_d2h_raw(dptr, dst, len)? };
                self.stats.shared_bytes += len as u64;
                Ok(Value::Nil)
            }
            "module_load" => self.module_load(args),
            "module_unload" => {
                let module = arg_u64(args, "module")?;
                if !self.modules.contains_key(&module) {
                    return Err(CuError::new(
                        cu::INVALID_HANDLE,
                        format!("module 0x{module:x} is not owned by this client"),
                    ));
                }
                self.backend()?.module_unload(module)?;
                self.modules.remove(&module);
                self.functions.retain(|_, (owner, _)| *owner != module);
                Ok(Value::Nil)
            }
            "function_get" => self.function_get(args),
            "graph_create" => self.graph_create(args),
            "graph_launch" => {
                let graph = self.owned_graph(args)?;
                let count = match proto::get(args, "count") {
                    None => 1,
                    Some(_) => arg_u32(args, "count")?,
                };
                if !(1..=MAX_GRAPH_REPLAYS).contains(&count) {
                    return Err(CuError::new(
                        cu::INVALID_VALUE,
                        format!("graph launch count must be 1..={MAX_GRAPH_REPLAYS}"),
                    ));
                }
                self.backend()?.graph_launch(graph, count)?;
                self.stats.launches += u64::from(count);
                Ok(Value::Nil)
            }
            "graph_destroy" => {
                let graph = self.owned_graph(args)?;
                self.backend()?.graph_destroy(graph)?;
                self.graphs.remove(&graph);
                Ok(Value::Nil)
            }
            "launch" => self.launch(args),
            other => Err(CuError::new(
                cu::NOT_SUPPORTED,
                format!("unknown op `{other}`"),
            )),
        }
    }

    fn hello(&mut self, args: &Value) -> CuResult<Value> {
        let abi = arg_u64(args, "abi")?;
        if abi != CLIENT_ABI {
            return Err(CuError::new(
                cu::NOT_SUPPORTED,
                format!("client ABI {abi} is not supported (server speaks {CLIENT_ABI})"),
            ));
        }
        self.greeted = true;
        let backend = self.backend()?.name();
        Ok(map(vec![
            ("server_kind", Value::from(SERVER_KIND)),
            ("protocol", Value::from(PROTOCOL)),
            ("backend", Value::from(backend)),
            ("abi", Value::from(CLIENT_ABI)),
            ("max_chunk", Value::from(proto::MAX_CHUNK_BYTES as u64)),
            ("shared_host_memory", Value::from(true)),
            (
                "features",
                Value::Array(
                    self.backend()?
                        .features()
                        .into_iter()
                        .map(Value::from)
                        .collect(),
                ),
            ),
        ]))
    }

    /// Create a shared host segment. Its descriptor follows the reply.
    fn host_alloc(&mut self, args: &Value) -> CuResult<Value> {
        if self.segments.len() >= MAX_SEGMENTS {
            return Err(CuError::new(
                cu::OUT_OF_MEMORY,
                format!("a client may hold at most {MAX_SEGMENTS} host segments"),
            ));
        }
        let size = usize::try_from(arg_u64(args, "size")?)
            .map_err(|_| CuError::new(cu::INVALID_VALUE, "segment size exceeds usize"))?;
        let segment = Segment::create(size)?;
        self.next_segment += 1;
        let handle = self.next_segment;
        let reply = map(vec![
            ("handle", Value::from(handle)),
            ("size", Value::from(segment.len() as u64)),
        ]);
        self.pending_fd = Some(segment.fd());
        self.segments.insert(handle, segment);
        Ok(reply)
    }

    /// Validate `{dptr, handle, offset, len}` and return the host pointer.
    fn segment_range(&self, args: &Value) -> CuResult<(u64, *mut u8, usize)> {
        let handle = arg_u64(args, "handle")?;
        let len = arg_u64(args, "len")?;
        if len > MAX_TRANSFER_BYTES {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!("copy of {len} bytes exceeds the {MAX_TRANSFER_BYTES} byte cap"),
            ));
        }
        let segment = self.segments.get(&handle).ok_or_else(|| {
            CuError::new(cu::INVALID_VALUE, format!("unknown host segment {handle}"))
        })?;
        let host = segment.range(arg_u64(args, "offset")?, len)?;
        Ok((arg_u64(args, "dptr")?, host, len as usize))
    }

    /// Uploads arrive in chunks and reach the backend as ONE copy at the
    /// client's pointer, so the backend never sees a partially written range.
    fn h2d(&mut self, args: &Value) -> CuResult<Value> {
        let dptr = arg_u64(args, "dptr")?;
        let total = arg_u64(args, "total")?;
        let offset = arg_u64(args, "offset")?;
        let data = arg_bin(args, "data")?;
        if total > MAX_TRANSFER_BYTES {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!("upload of {total} bytes exceeds the {MAX_TRANSFER_BYTES} byte cap"),
            ));
        }
        let total = total as usize;
        if offset == 0 {
            self.upload = Some(Upload {
                dptr,
                data: Vec::with_capacity(total),
                total,
            });
        }
        let upload = self
            .upload
            .as_mut()
            .filter(|u| u.dptr == dptr && u.total == total && u.data.len() as u64 == offset)
            .ok_or_else(|| {
                CuError::new(
                    cu::INVALID_VALUE,
                    "upload chunk does not continue the upload in progress",
                )
            })?;
        if upload.data.len() + data.len() > total {
            self.upload = None;
            return Err(CuError::new(
                cu::INVALID_VALUE,
                "upload chunks exceed the declared total",
            ));
        }
        upload.data.extend_from_slice(data);
        self.stats.socket_bytes += data.len() as u64;
        if upload.data.len() == total {
            let upload = self.upload.take().expect("upload in progress");
            self.backend()?.copy_h2d(upload.dptr, &upload.data)?;
        }
        Ok(Value::Nil)
    }

    /// The backend is read once per download; later chunks come from that copy.
    fn d2h(&mut self, args: &Value) -> CuResult<Value> {
        let dptr = arg_u64(args, "dptr")?;
        let total = arg_u64(args, "total")?;
        let offset = arg_u64(args, "offset")? as usize;
        let len = arg_u64(args, "len")? as usize;
        if total > MAX_TRANSFER_BYTES {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!("download of {total} bytes exceeds the {MAX_TRANSFER_BYTES} byte cap"),
            ));
        }
        let total = total as usize;
        if offset == 0 {
            let data = self.backend()?.copy_d2h(dptr, total)?;
            self.download = Some(Download { dptr, data });
        }
        let download = self
            .download
            .as_ref()
            .filter(|d| d.dptr == dptr && d.data.len() == total)
            .ok_or_else(|| {
                CuError::new(
                    cu::INVALID_VALUE,
                    "download chunk does not continue the download in progress",
                )
            })?;
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= total && len <= proto::MAX_CHUNK_BYTES)
            .ok_or_else(|| CuError::new(cu::INVALID_VALUE, "download chunk is out of range"))?;
        let chunk = Value::Binary(download.data[offset..end].to_vec());
        self.stats.socket_bytes += (end - offset) as u64;
        if end == total {
            self.download = None;
        }
        Ok(chunk)
    }

    fn module_load(&mut self, args: &Value) -> CuResult<Value> {
        let image = arg_bin(args, "image")?;
        let options = match proto::get(args, "opt_level").and_then(Value::as_u64) {
            Some(level) => JitOptions {
                opt_level: level.min(4) as u32,
            },
            None => JitOptions::default(),
        };
        let prepared = module::prepare(image, options, &self.state.ptxas)?;
        let handle = self.backend()?.module_load(&prepared.cubin)?;
        self.modules.insert(
            handle,
            LoadedModule {
                cubin: prepared.cubin,
            },
        );
        Ok(map(vec![
            ("module", Value::from(handle)),
            ("compiled", Value::from(prepared.compiled)),
            ("log", Value::from(prepared.log)),
        ]))
    }

    fn function_get(&mut self, args: &Value) -> CuResult<Value> {
        let module = arg_u64(args, "module")?;
        let name = arg_str(args, "name")?;
        let loaded = self.modules.get(&module).ok_or_else(|| {
            CuError::new(
                cu::INVALID_HANDLE,
                format!("module 0x{module:x} is not owned by this client"),
            )
        })?;
        // cuLaunchKernel carries no parameter sizes; they come from the cubin.
        let abi = cuda_driverkit::cubin::kernel_abi(&loaded.cubin, name)
            .map_err(|e| CuError::new(cu::NOT_FOUND, format!("kernel `{name}`: {}", e.0)))?;
        let sizes: Vec<u32> = abi.params.iter().map(|p| p.size as u32).collect();
        // Offsets let the client split a packed parameter block (the `extra`
        // form of cuLaunchKernel and the CUDA 3.x cuParamSet* protocol).
        let offsets: Vec<u32> = abi.params.iter().map(|p| p.offset as u32).collect();
        let function = self.backend()?.function_get(module, name)?;
        self.functions.insert(function, (module, sizes.clone()));
        Ok(map(vec![
            ("function", Value::from(function)),
            (
                "param_sizes",
                Value::Array(sizes.into_iter().map(Value::from).collect()),
            ),
            (
                "param_offsets",
                Value::Array(offsets.into_iter().map(Value::from).collect()),
            ),
        ]))
    }

    /// Validate one launch description against the function's parameter layout.
    fn checked_launch(&self, args: &Value) -> CuResult<GraphLaunch> {
        let function = arg_u64(args, "function")?;
        let (_, sizes) = self.functions.get(&function).ok_or_else(|| {
            CuError::new(
                cu::INVALID_HANDLE,
                format!("function 0x{function:x} is not owned by this client"),
            )
        })?;
        let grid = arg_dims(args, "grid")?;
        let block = arg_dims(args, "block")?;
        if grid.iter().chain(block.iter()).any(|d| *d == 0) {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                "grid and block dimensions must be non-zero",
            ));
        }
        let items = proto::get(args, "params")
            .and_then(Value::as_array)
            .ok_or_else(|| CuError::new(cu::INVALID_VALUE, "`params` must be an array"))?;
        if items.len() != sizes.len() {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!(
                    "kernel takes {} parameters, {} were sent",
                    sizes.len(),
                    items.len()
                ),
            ));
        }
        let mut params = Vec::with_capacity(items.len());
        for (index, (item, size)) in items.iter().zip(sizes).enumerate() {
            match item {
                Value::Binary(bytes) if bytes.len() == *size as usize => params.push(bytes.clone()),
                _ => {
                    return Err(CuError::new(
                        cu::INVALID_VALUE,
                        format!("parameter {index} must be {size} bytes"),
                    ))
                }
            }
        }
        Ok(GraphLaunch {
            function,
            grid,
            block,
            shared_mem_bytes: arg_u32(args, "shared_mem_bytes")?,
            params,
        })
    }

    fn launch(&mut self, args: &Value) -> CuResult<Value> {
        let launch = self.checked_launch(args)?;
        self.backend()?.launch(
            launch.function,
            launch.grid,
            launch.block,
            launch.shared_mem_bytes,
            &launch.params,
        )?;
        self.stats.launches += 1;
        Ok(Value::Nil)
    }

    /// Record an ordered list of launches once; `graph_launch` replays it.
    fn graph_create(&mut self, args: &Value) -> CuResult<Value> {
        let items = proto::get(args, "launches")
            .and_then(Value::as_array)
            .filter(|items| !items.is_empty() && items.len() <= MAX_GRAPH_LAUNCHES)
            .ok_or_else(|| {
                CuError::new(
                    cu::INVALID_VALUE,
                    format!("`launches` must be an array of 1..={MAX_GRAPH_LAUNCHES} launches"),
                )
            })?;
        let mut launches = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            launches.push(self.checked_launch(item).map_err(|e| {
                CuError::new(e.code, format!("graph launch {index}: {}", e.reason))
            })?);
        }
        let graph = self.backend()?.graph_create(&launches)?;
        self.graphs.insert(graph);
        Ok(map(vec![
            ("graph", Value::from(graph)),
            ("launches", Value::from(launches.len() as u64)),
        ]))
    }

    fn owned_graph(&self, args: &Value) -> CuResult<u64> {
        let graph = arg_u64(args, "graph")?;
        if !self.graphs.contains(&graph) {
            return Err(CuError::new(
                cu::INVALID_HANDLE,
                format!("graph 0x{graph:x} is not owned by this client"),
            ));
        }
        Ok(graph)
    }

    /// Return everything the client still holds. Errors are logged, not fatal:
    /// the client is already gone.
    fn release_all(&mut self) {
        let Ok(mut backend) = self.state.backend.lock() else {
            return;
        };
        // Graphs first: they pin the buffers and modules they launch from.
        for graph in std::mem::take(&mut self.graphs) {
            if let Err(e) = backend.graph_destroy(graph) {
                eprintln!(
                    "cuda_adapter_server: cleanup graph 0x{graph:x}: {}",
                    e.reason
                );
            }
        }
        for module in std::mem::take(&mut self.modules).into_keys() {
            if let Err(e) = backend.module_unload(module) {
                eprintln!(
                    "cuda_adapter_server: cleanup module 0x{module:x}: {}",
                    e.reason
                );
            }
        }
        for dptr in std::mem::take(&mut self.allocations) {
            if let Err(e) = backend.mem_free(dptr) {
                eprintln!(
                    "cuda_adapter_server: cleanup allocation 0x{dptr:x}: {}",
                    e.reason
                );
            }
        }
        for ctx in std::mem::take(&mut self.contexts) {
            if let Err(e) = backend.ctx_destroy(ctx) {
                eprintln!(
                    "cuda_adapter_server: cleanup context 0x{ctx:x}: {}",
                    e.reason
                );
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// Serve one connection until the peer closes it. A second client that
/// arrives while one is active is told the adapter is busy and disconnected.
pub fn serve_connection(state: Arc<ServerState>, mut stream: impl FdChannel) {
    if state.busy.swap(true, Ordering::SeqCst) {
        if let Ok(Some(body)) = proto::read_frame(&mut stream) {
            let seq = proto::decode_request(&body).map(|r| r.seq).unwrap_or(0);
            let busy = CuError::new(cu::UNKNOWN, "adapter busy: another client holds the card");
            let _ = proto::write_frame(&mut stream, &proto::encode_err(seq, &busy));
        }
        return;
    }
    let mut session = Session::new(state.clone());
    loop {
        let body = match proto::read_frame(&mut stream) {
            Ok(Some(body)) => body,
            Ok(None) => break,
            Err(e) => {
                eprintln!("cuda_adapter_server: client read failed: {e}");
                break;
            }
        };
        let reply = match proto::decode_request(&body) {
            Ok(request) => match session.dispatch(&request.op, &request.args) {
                Ok(result) => proto::encode_ok(request.seq, result),
                Err(error) => {
                    eprintln!(
                        "cuda_adapter_server: {} -> CUDA error {}: {}",
                        request.op, error.code, error.reason
                    );
                    proto::encode_err(request.seq, &error)
                }
            },
            Err(error) => proto::encode_err(0, &error),
        };
        if let Err(e) = proto::write_frame(&mut stream, &reply) {
            eprintln!("cuda_adapter_server: client write failed: {e}");
            break;
        }
        if let Some(fd) = session.pending_fd.take() {
            if let Err(e) = stream.send_fd(fd) {
                eprintln!("cuda_adapter_server: could not pass a segment descriptor: {e}");
                break;
            }
        }
    }
    let stats = session.stats;
    drop(session);
    println!(
        "cuda_adapter_server: client done: {} launch(es), {} copy bytes over the socket, {} through shared segments",
        stats.launches, stats.socket_bytes, stats.shared_bytes
    );
    state.busy.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;

    use super::*;
    use crate::backend::Loopback;

    fn state() -> Arc<ServerState> {
        ServerState::new(
            Box::new(Loopback::default()),
            PtxasLocator::Explicit(PathBuf::from("/nonexistent/ptxas")),
        )
    }

    /// Real sm_120 cubins from the driverkit test assets. They are not tracked
    /// in git, so a clean checkout has none: the test then says so and skips.
    fn cubin(name: &str) -> Option<Vec<u8>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/cuda-driverkit/test_assets/cubins")
            .join(name);
        match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(_) => {
                eprintln!("SKIPPED: test cubin {} is not installed", path.display());
                None
            }
        }
    }

    fn greeted(state: &Arc<ServerState>) -> Session {
        let mut session = Session::new(state.clone());
        session
            .dispatch("hello", &map(vec![("abi", Value::from(CLIENT_ABI))]))
            .unwrap();
        session
            .dispatch("init", &map(vec![("flags", Value::from(0))]))
            .unwrap();
        session
    }

    fn u64_of(value: &Value, key: &str) -> u64 {
        proto::get(value, key).and_then(Value::as_u64).unwrap()
    }

    fn free_bytes(session: &mut Session) -> u64 {
        u64_of(&session.dispatch("mem_info", &map(vec![])).unwrap(), "free")
    }

    #[test]
    fn hello_is_required_and_checks_the_abi() {
        let state = state();
        let mut session = Session::new(state);
        assert_eq!(
            session.dispatch("init", &map(vec![])).unwrap_err().code,
            cu::NOT_INITIALIZED
        );
        assert_eq!(
            session
                .dispatch("hello", &map(vec![("abi", Value::from(99))]))
                .unwrap_err()
                .code,
            cu::NOT_SUPPORTED
        );
        let hello = session
            .dispatch("hello", &map(vec![("abi", Value::from(CLIENT_ABI))]))
            .unwrap();
        assert_eq!(
            proto::get(&hello, "server_kind").and_then(Value::as_str),
            Some(SERVER_KIND)
        );
        assert_eq!(
            proto::get(&hello, "backend").and_then(Value::as_str),
            Some("loopback")
        );
    }

    #[test]
    fn function_get_reports_the_cubin_parameter_layout() {
        let Some(image) = cubin("matmul_fp16_mma.cubin") else {
            return;
        };
        let state = state();
        let mut session = greeted(&state);
        let image = Value::Binary(image);
        let loaded = session
            .dispatch("module_load", &map(vec![("image", image)]))
            .unwrap();
        let module = u64_of(&loaded, "module");
        let reply = session
            .dispatch(
                "function_get",
                &map(vec![
                    ("module", Value::from(module)),
                    ("name", Value::from("matmul_fp16_mma")),
                ]),
            )
            .unwrap();
        let list = |key: &str| -> Vec<u64> {
            proto::get(&reply, key)
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect()
        };
        // Three device pointers, then three 32-bit sizes.
        assert_eq!(list("param_sizes"), vec![8, 8, 8, 4, 4, 4]);
        assert_eq!(list("param_offsets"), vec![0, 8, 16, 24, 28, 32]);
        let missing = session
            .dispatch(
                "function_get",
                &map(vec![
                    ("module", Value::from(module)),
                    ("name", Value::from("nope")),
                ]),
            )
            .unwrap_err();
        assert_eq!(missing.code, cu::NOT_FOUND);
    }

    #[test]
    fn launch_checks_parameter_count_and_width_before_the_backend() {
        let Some(image) = cubin("vecadd_macos.cubin") else {
            return;
        };
        let state = state();
        let mut session = greeted(&state);
        let loaded = session
            .dispatch("module_load", &map(vec![("image", Value::Binary(image))]))
            .unwrap();
        let function = u64_of(
            &session
                .dispatch(
                    "function_get",
                    &map(vec![
                        ("module", Value::from(u64_of(&loaded, "module"))),
                        ("name", Value::from("vecadd_macos")),
                    ]),
                )
                .unwrap(),
            "function",
        );
        let dims = |x: u32| Value::Array(vec![Value::from(x), Value::from(1), Value::from(1)]);
        let launch = |session: &mut Session, params: Vec<Value>| {
            session.dispatch(
                "launch",
                &map(vec![
                    ("function", Value::from(function)),
                    ("grid", dims(1)),
                    ("block", dims(4)),
                    ("shared_mem_bytes", Value::from(0)),
                    ("params", Value::Array(params)),
                ]),
            )
        };
        let p8 = || Value::Binary(vec![0u8; 8]);
        let too_few = launch(&mut session, vec![p8(), p8(), p8()]).unwrap_err();
        assert!(too_few.reason.contains("takes 4 parameters"));
        // The fourth parameter is a u32: an 8-byte value must be refused, not truncated.
        let wide = launch(&mut session, vec![p8(), p8(), p8(), p8()]).unwrap_err();
        assert!(wide.reason.contains("parameter 3 must be 4 bytes"));
    }

    #[test]
    fn chunked_upload_reaches_the_backend_once_complete_and_rejects_gaps() {
        let state = state();
        let mut session = greeted(&state);
        let dptr = session
            .dispatch("mem_alloc", &map(vec![("size", Value::from(8))]))
            .unwrap()
            .as_u64()
            .unwrap();
        let chunk = |offset: u64, data: &[u8]| {
            map(vec![
                ("dptr", Value::from(dptr)),
                ("total", Value::from(8)),
                ("offset", Value::from(offset)),
                ("data", Value::Binary(data.to_vec())),
            ])
        };
        let read = |session: &mut Session| {
            let args = map(vec![
                ("dptr", Value::from(dptr)),
                ("total", Value::from(8)),
                ("offset", Value::from(0)),
                ("len", Value::from(8)),
            ]);
            match session.dispatch("d2h", &args).unwrap() {
                Value::Binary(bytes) => bytes,
                other => panic!("expected bytes, got {other:?}"),
            }
        };
        session.dispatch("h2d", &chunk(0, &[1, 2, 3, 4])).unwrap();
        assert_eq!(
            read(&mut session),
            vec![0; 8],
            "a partial upload must not be visible"
        );
        session.dispatch("h2d", &chunk(4, &[5, 6, 7, 8])).unwrap();
        assert_eq!(read(&mut session), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // A chunk that skips ahead does not continue any upload.
        session.dispatch("h2d", &chunk(0, &[9, 9])).unwrap();
        assert_eq!(
            session
                .dispatch("h2d", &chunk(6, &[9, 9]))
                .unwrap_err()
                .code,
            cu::INVALID_VALUE
        );
    }

    #[test]
    fn a_client_cannot_free_what_it_does_not_own_and_a_dropped_session_returns_everything() {
        let Some(image) = cubin("vecadd_macos.cubin") else {
            return;
        };
        let state = state();
        let mut first = greeted(&state);
        let total = free_bytes(&mut first);
        let dptr = first
            .dispatch("mem_alloc", &map(vec![("size", Value::from(4096))]))
            .unwrap()
            .as_u64()
            .unwrap();
        first
            .dispatch("module_load", &map(vec![("image", Value::Binary(image))]))
            .unwrap();
        first
            .dispatch(
                "ctx_create",
                &map(vec![("device", Value::from(0)), ("flags", Value::from(0))]),
            )
            .unwrap();

        let mut second = greeted(&state);
        assert_eq!(
            second
                .dispatch("mem_free", &map(vec![("dptr", Value::from(dptr))]))
                .unwrap_err()
                .code,
            cu::INVALID_VALUE
        );
        assert_eq!(free_bytes(&mut second), total - 4096);
        drop(first);
        assert_eq!(
            free_bytes(&mut second),
            total,
            "the dropped client's allocation was returned"
        );
    }

    #[test]
    fn shared_segment_copies_carry_no_payload_and_stay_inside_the_segment() {
        let state = state();
        let mut session = greeted(&state);
        let reply = session
            .dispatch("host_alloc", &map(vec![("size", Value::from(1000))]))
            .unwrap();
        let handle = u64_of(&reply, "handle");
        assert_eq!(u64_of(&reply, "size"), crate::shm::SEGMENT_ALIGN as u64);
        assert!(
            session.pending_fd.take().is_some(),
            "the descriptor is queued for the reply"
        );
        let dptr = session
            .dispatch("mem_alloc", &map(vec![("size", Value::from(64))]))
            .unwrap()
            .as_u64()
            .unwrap();
        let copy = |op: &str, offset: u64, len: u64| {
            (
                op.to_string(),
                map(vec![
                    ("dptr", Value::from(dptr)),
                    ("handle", Value::from(handle)),
                    ("offset", Value::from(offset)),
                    ("len", Value::from(len)),
                ]),
            )
        };
        // Write through the server's mapping, as the client would through its own.
        let base = session.segments[&handle].range(0, 128).unwrap();
        for i in 0..64u8 {
            // SAFETY: inside the 16 KiB segment.
            unsafe { *base.add(i as usize) = i };
        }
        let (op, args) = copy("h2d_shm", 0, 64);
        session.dispatch(&op, &args).unwrap();
        let (op, args) = copy("d2h_shm", 64, 64);
        session.dispatch(&op, &args).unwrap();
        for i in 0..64u8 {
            // SAFETY: inside the segment.
            assert_eq!(unsafe { *base.add(64 + i as usize) }, i);
        }
        let (op, args) = copy("h2d_shm", crate::shm::SEGMENT_ALIGN as u64 - 8, 16);
        assert_eq!(
            session.dispatch(&op, &args).unwrap_err().code,
            cu::INVALID_VALUE
        );
        assert_eq!(
            session.stats,
            TransferStats {
                socket_bytes: 0,
                shared_bytes: 128,
                launches: 0
            }
        );
        session
            .dispatch("host_free", &map(vec![("handle", Value::from(handle))]))
            .unwrap();
        let (op, args) = copy("h2d_shm", 0, 8);
        assert_eq!(
            session.dispatch(&op, &args).unwrap_err().code,
            cu::INVALID_VALUE
        );
    }

    #[test]
    fn a_second_connection_is_told_the_adapter_is_busy() {
        let state = state();
        let (mut first_client, first_server) = UnixStream::pair().unwrap();
        let (mut second_client, second_server) = UnixStream::pair().unwrap();
        let request = |seq: u64| {
            let mut body = Vec::new();
            let value = map(vec![
                ("op", Value::from("hello")),
                ("seq", Value::from(seq)),
                ("args", map(vec![("abi", Value::from(CLIENT_ABI))])),
            ]);
            rmpv::encode::write_value(&mut body, &value).unwrap();
            body
        };
        let reply_status = |stream: &mut UnixStream| {
            let body = proto::read_frame(stream).unwrap().unwrap();
            let value = rmpv::decode::read_value(&mut &body[..]).unwrap();
            (
                proto::get(&value, "status")
                    .and_then(Value::as_str)
                    .unwrap()
                    .to_string(),
                proto::get(&value, "error")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            )
        };
        let first = {
            let state = state.clone();
            std::thread::spawn(move || serve_connection(state, first_server))
        };
        proto::write_frame(&mut first_client, &request(1)).unwrap();
        assert_eq!(reply_status(&mut first_client).0, "ok");

        proto::write_frame(&mut second_client, &request(1)).unwrap();
        serve_connection(state.clone(), second_server);
        let (status, error) = reply_status(&mut second_client);
        assert_eq!(status, "error");
        assert!(error.contains("busy"));

        // Once the first client leaves, the adapter is free again.
        drop(first_client);
        first.join().unwrap();
        let (mut third_client, third_server) = UnixStream::pair().unwrap();
        let third = std::thread::spawn(move || serve_connection(state, third_server));
        proto::write_frame(&mut third_client, &request(1)).unwrap();
        assert_eq!(reply_status(&mut third_client).0, "ok");
        drop(third_client);
        third.join().unwrap();
    }
}
