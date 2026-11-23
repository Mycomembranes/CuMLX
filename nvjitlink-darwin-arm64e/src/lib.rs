// SPDX-License-Identifier: Apache-2.0
//
//! Darwin arm64e nvJitLink wrapper dylib.
//!
//! This crate builds to `libnvJitLink.dylib` and exports the same C ABI
//! surface as NVIDIA's nvJitLink. It does not reimplement the linker; it
//! forwards every call to a backend library loaded from a separate path.
#![allow(non_snake_case)]
//!
//! The point is portability:
//! - `NVJITLINK_PATH` can point at this wrapper dylib.
//! - `NVJITLINK_BACKEND_PATH` can point at the actual backend dylib if one
//!   exists on the machine.
//! - The wrapper does not try to load itself recursively. If no backend is
//!   provided, it fails cleanly and the caller can fall back to PTX.

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;
use std::ffi::{c_char, c_void};
use std::fs;
use std::path::PathBuf;
use std::ptr;
use thiserror::Error;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NvJitLinkHandle(pub *mut c_void);

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum NvJitLinkResult {
    Success = 0,
    UnrecognizedOption = 1,
    MissingArch = 2,
    InvalidInput = 3,
    PtxCompile = 4,
    NvvmCompile = 5,
    Internal = 6,
    Threadpool = 7,
    UnrecognizedInput = 8,
    Finalize = 9,
    NullInput = 10,
    IncompatibleOptions = 11,
    IncorrectInputType = 12,
    ArchMismatch = 13,
    OutdatedLibrary = 14,
    MissingFatbin = 15,
    UnrecognizedArch = 16,
    UnsupportedArch = 17,
    LtoNotEnabled = 18,
}

impl NvJitLinkResult {
    const fn internal_error() -> Self {
        NvJitLinkResult::Internal
    }
}

#[derive(Debug, Error)]
enum BackendLoadError {
    #[error("nvJitLink backend not found. Tried:\n  {tried}")]
    NotFound { tried: String },
    #[error("nvJitLink backend symbol missing: {symbol}: {source}")]
    Symbol {
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },
}

struct Backend {
    _lib: Library,
    create:
        unsafe extern "C" fn(*mut NvJitLinkHandle, u32, *const *const c_char) -> NvJitLinkResult,
    destroy: unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult,
    add_data: unsafe extern "C" fn(
        NvJitLinkHandle,
        u32,
        *const c_void,
        usize,
        *const c_char,
    ) -> NvJitLinkResult,
    complete: unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult,
    get_linked_cubin_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_linked_cubin: unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult,
    get_error_log_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_error_log: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
    get_info_log_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_info_log: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
    version: Option<unsafe extern "C" fn(*mut u32, *mut u32) -> NvJitLinkResult>,
}

// The wrapper is a plain FFI bridge. Calls are forwarded through a shared
// backend handle loaded once at process start.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

struct WrapperLinker {
    backend_handle: NvJitLinkHandle,
}

static BACKEND: OnceCell<Result<&'static Backend, &'static BackendLoadError>> = OnceCell::new();

fn backend() -> Result<&'static Backend, &'static BackendLoadError> {
    match BACKEND.get_or_init(load_backend) {
        Ok(backend) => Ok(*backend),
        Err(err) => Err(*err),
    }
}

fn load_backend() -> Result<&'static Backend, &'static BackendLoadError> {
    let mut tried = Vec::new();
    let lib = open_backend_library(&mut tried).ok_or_else(|| {
        let err: &'static BackendLoadError = Box::leak(Box::new(BackendLoadError::NotFound {
            tried: tried.join("\n  "),
        }));
        err
    })?;

    unsafe {
        let create = resolve(&lib, "nvJitLinkCreate")?;
        let destroy = resolve(&lib, "nvJitLinkDestroy")?;
        let add_data = resolve(&lib, "nvJitLinkAddData")?;
        let complete = resolve(&lib, "nvJitLinkComplete")?;
        let get_linked_cubin_size = resolve(&lib, "nvJitLinkGetLinkedCubinSize")?;
        let get_linked_cubin = resolve(&lib, "nvJitLinkGetLinkedCubin")?;
        let get_error_log_size = resolve(&lib, "nvJitLinkGetErrorLogSize")?;
        let get_error_log = resolve(&lib, "nvJitLinkGetErrorLog")?;
        let get_info_log_size = resolve(&lib, "nvJitLinkGetInfoLogSize")?;
        let get_info_log = resolve(&lib, "nvJitLinkGetInfoLog")?;
        let version = resolve_optional(&lib, "nvJitLinkVersion");
        let backend = Backend {
            create,
            destroy,
            add_data,
            complete,
            get_linked_cubin_size,
            get_linked_cubin,
            get_error_log_size,
            get_error_log,
            get_info_log_size,
            get_info_log,
            version,
            _lib: lib,
        };
        Ok(Box::leak(Box::new(backend)))
    }
}

unsafe fn resolve<T: Copy>(
    lib: &Library,
    name: &'static str,
) -> Result<T, &'static BackendLoadError> {
    let sym: Symbol<T> = unsafe { lib.get(name.as_bytes()) }.map_err(|source| {
        let err: &'static BackendLoadError = Box::leak(Box::new(BackendLoadError::Symbol {
            symbol: name,
            source,
        }));
        err
    })?;
    Ok(unsafe { *sym.into_raw() })
}

unsafe fn resolve_optional<T: Copy>(lib: &Library, name: &'static str) -> Option<T> {
    let sym: Result<Symbol<T>, _> = unsafe { lib.get(name.as_bytes()) };
    sym.ok().map(|s| unsafe { *s.into_raw() })
}

fn open_backend_library(tried: &mut Vec<String>) -> Option<Library> {
    let wrapper_paths = [
        "NVJITLINK_PATH",
        "NVJITLINK_DARWIN_PATH",
        "CUDA_OXIDE_NVJITLINK_PATH",
    ]
    .into_iter()
    .filter_map(|var| std::env::var(var).ok())
    .collect::<Vec<_>>();

    for var in [
        "NVJITLINK_BACKEND_PATH",
        "LIBNVJITLINK_PATH",
        "CUDA_OXIDE_NVJITLINK_BACKEND_PATH",
    ] {
        if let Ok(p) = std::env::var(var) {
            let path = PathBuf::from(&p);
            tried.push(format!("{var}={}", path.display()));
            let skip_self = should_skip_self_reference(&path, &wrapper_paths);
            if skip_self {
                tried.push(format!("{var}=<skipped-self>"));
                continue;
            }
            if let Ok(lib) = unsafe { Library::new(&path) } {
                return Some(lib);
            }
        }
    }

    None
}

fn should_skip_self_reference(path: &PathBuf, wrapper_paths: &[String]) -> bool {
    wrapper_paths.iter().any(|wrapper| {
        path == &PathBuf::from(wrapper)
            || fs::canonicalize(path).ok() == fs::canonicalize(wrapper).ok()
    })
}

fn wrapper_from_handle(handle: NvJitLinkHandle) -> Option<&'static mut WrapperLinker> {
    if handle.0.is_null() {
        return None;
    }
    Some(unsafe { &mut *(handle.0 as *mut WrapperLinker) })
}

fn backend_call<F>(f: F) -> NvJitLinkResult
where
    F: FnOnce(&Backend) -> NvJitLinkResult,
{
    match backend() {
        Ok(backend) => f(backend),
        Err(_) => NvJitLinkResult::internal_error(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkVersion(major: *mut u32, minor: *mut u32) -> NvJitLinkResult {
    match backend() {
        Ok(backend) => match backend.version {
            Some(v) => unsafe { v(major, minor) },
            None => {
                if !major.is_null() {
                    unsafe { *major = 0 };
                }
                if !minor.is_null() {
                    unsafe { *minor = 0 };
                }
                NvJitLinkResult::Success
            }
        },
        Err(_) => NvJitLinkResult::internal_error(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkCreate(
    handle: *mut NvJitLinkHandle,
    num_options: u32,
    options: *const *const c_char,
) -> NvJitLinkResult {
    if handle.is_null() {
        return NvJitLinkResult::NullInput;
    }
    backend_call(|backend| {
        let mut backend_handle = NvJitLinkHandle(ptr::null_mut());
        let r = unsafe { (backend.create)(&mut backend_handle, num_options, options) };
        if r != NvJitLinkResult::Success {
            return r;
        }
        let wrapper = Box::new(WrapperLinker { backend_handle });
        let wrapper_ptr = Box::into_raw(wrapper) as *mut c_void;
        unsafe {
            *handle = NvJitLinkHandle(wrapper_ptr);
        }
        NvJitLinkResult::Success
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkDestroy(handle: *mut NvJitLinkHandle) -> NvJitLinkResult {
    if handle.is_null() {
        return NvJitLinkResult::NullInput;
    }
    let raw = unsafe { (*handle).0 };
    if raw.is_null() {
        return NvJitLinkResult::Success;
    }
    let wrapper = unsafe { Box::from_raw(raw as *mut WrapperLinker) };
    let mut backend_handle = wrapper.backend_handle;
    let result = backend_call(|backend| unsafe { (backend.destroy)(&mut backend_handle) });
    unsafe {
        (*handle).0 = ptr::null_mut();
    }
    drop(wrapper);
    result
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkAddData(
    handle: NvJitLinkHandle,
    kind: u32,
    data: *const c_void,
    size: usize,
    name: *const c_char,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe {
        (backend.add_data)(wrapper.backend_handle, kind, data, size, name)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkComplete(handle: NvJitLinkHandle) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.complete)(wrapper.backend_handle) })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkGetLinkedCubinSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.get_linked_cubin_size)(wrapper.backend_handle, size) })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkGetLinkedCubin(
    handle: NvJitLinkHandle,
    cubin: *mut c_void,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.get_linked_cubin)(wrapper.backend_handle, cubin) })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkGetErrorLogSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.get_error_log_size)(wrapper.backend_handle, size) })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkGetErrorLog(
    handle: NvJitLinkHandle,
    buf: *mut c_char,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.get_error_log)(wrapper.backend_handle, buf) })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkGetInfoLogSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.get_info_log_size)(wrapper.backend_handle, size) })
}

#[unsafe(no_mangle)]
pub extern "C" fn nvJitLinkGetInfoLog(
    handle: NvJitLinkHandle,
    buf: *mut c_char,
) -> NvJitLinkResult {
    let Some(wrapper) = wrapper_from_handle(handle) else {
        return NvJitLinkResult::NullInput;
    };
    backend_call(|backend| unsafe { (backend.get_info_log)(wrapper.backend_handle, buf) })
}

#[cfg(test)]
mod tests {
    use super::{open_backend_library, should_skip_self_reference};
    use std::path::PathBuf;

    #[test]
    fn backend_search_is_fail_fast_without_explicit_backend_path() {
        let mut tried = Vec::new();
        let _ = open_backend_library(&mut tried);
        assert!(tried.is_empty());
    }

    #[test]
    fn backend_search_skips_wrapper_self_reference() {
        let path = PathBuf::from("/tmp/libnvJitLink.dylib");
        let wrappers = vec!["/tmp/libnvJitLink.dylib".to_string()];
        assert!(should_skip_self_reference(&path, &wrappers));
    }
}
