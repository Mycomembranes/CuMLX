// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `cuda_adapter_server` — serves the CUDA Driver API to stand-in `libcuda`
//! forwarders over a private AF_UNIX socket. See `portkit/cuda-adapter/README.md`.

use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use cuda_adapter_server::backend::{Backend, Loopback};
use cuda_adapter_server::backend_driverkit::DriverKit;
use cuda_adapter_server::module::PtxasLocator;
use cuda_adapter_server::server::{serve_connection, ServerState};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_signal: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

const USAGE: &str = "usage: cuda_adapter_server --backend loopback|driverkit [--socket PATH] [--serve-state PATH]\n\
  --backend loopback   host memory only, no GPU (tests)\n\
  --backend driverkit  the served card, through the exact-serve launch endpoint\n\
  --serve-state PATH   driverkit only: the served card's serve-pids.<id>.json registry\n\
                       (default: $CUDA_OXIDE_EXACT_SERVE_STATE)\n\
  --socket PATH        AF_UNIX path (default $CUDA_ADAPTER_SOCKET or /tmp/cuda-adapter-<uid>.sock)";

fn default_socket() -> PathBuf {
    if let Some(path) = std::env::var_os("CUDA_ADAPTER_SOCKET").filter(|v| !v.is_empty()) {
        return PathBuf::from(path);
    }
    // SAFETY: geteuid has no preconditions.
    PathBuf::from(format!("/tmp/cuda-adapter-{}.sock", unsafe {
        libc::geteuid()
    }))
}

fn main() -> Result<()> {
    let mut backend_name: Option<String> = None;
    let mut socket = default_socket();
    let mut serve_state: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => backend_name = Some(args.next().context("--backend needs a value")?),
            "--socket" => socket = PathBuf::from(args.next().context("--socket needs a value")?),
            "--serve-state" => {
                serve_state = Some(PathBuf::from(
                    args.next().context("--serve-state needs a value")?,
                ))
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown argument `{other}`\n{USAGE}"),
        }
    }
    // The backend is always named explicitly: nothing reaches a card by default.
    let backend: Box<dyn Backend> = match backend_name.as_deref() {
        Some("loopback") => {
            if serve_state.is_some() {
                bail!("--serve-state applies to the driverkit backend only");
            }
            Box::new(Loopback::default())
        }
        Some("driverkit") => {
            // The C ABI is a client of the exact-serve launch endpoint. The
            // environment is set before any thread exists; values the operator
            // already provided are left alone.
            if std::env::var_os("CUDA_OXIDE_CUDA_BACKEND").is_none() {
                // SAFETY: single-threaded at this point.
                unsafe { std::env::set_var("CUDA_OXIDE_CUDA_BACKEND", "pciserver") };
            }
            if let Some(path) = &serve_state {
                if !path.is_absolute() {
                    bail!("--serve-state must be an absolute path");
                }
                // SAFETY: single-threaded at this point.
                unsafe { std::env::set_var("CUDA_OXIDE_EXACT_SERVE_STATE", path) };
            }
            if std::env::var_os("CUDA_OXIDE_EXACT_SERVE_STATE").is_none_or(|v| v.is_empty()) {
                bail!(
                    "the driverkit backend needs the served card's registry: pass --serve-state \
                     or set CUDA_OXIDE_EXACT_SERVE_STATE"
                );
            }
            // Connect and authenticate now, so a missing or unqualified stack is
            // reported here and not to the first client. This opens the launch
            // socket only; it allocates nothing and launches nothing.
            let mut backend = DriverKit::default();
            if let Err(error) = backend.init() {
                bail!(
                    "cannot reach a qualified shared-CUDA stack ({}); is the card served with --shared-cuda?",
                    error.reason
                );
            }
            Box::new(backend)
        }
        Some(other) => bail!("unknown backend `{other}`\n{USAGE}"),
        None => bail!("--backend is required\n{USAGE}"),
    };
    let backend_label = backend.name();

    let listener = bind(&socket)?;
    listener.set_nonblocking(true)?;
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
    }
    let state = ServerState::new(backend, PtxasLocator::from_env());
    println!(
        "cuda_adapter_server listening on unix://{} backend={backend_label}",
        socket.display()
    );

    let mut workers = Vec::new();
    while !STOP.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Err(e) = admit(&stream) {
                    eprintln!("cuda_adapter_server: refused a connection: {e}");
                    continue;
                }
                stream.set_nonblocking(false)?;
                let state = state.clone();
                workers.push(std::thread::spawn(move || serve_connection(state, stream)));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100))
            }
            Err(e) => {
                eprintln!("cuda_adapter_server: accept failed: {e}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        workers.retain(|w| !w.is_finished());
    }
    // Stop accepting, then let any request in flight finish: a backend call is
    // never interrupted. Workers end when their client disconnects.
    drop(listener);
    let _ = std::fs::remove_file(&socket);
    println!(
        "cuda_adapter_server stopping; waiting for {} client(s) to finish",
        workers.len()
    );
    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}

/// Bind a 0600 socket. A stale socket file is replaced only when nothing answers on it.
fn bind(socket: &PathBuf) -> Result<UnixListener> {
    if socket.exists() {
        if UnixStream::connect(socket).is_ok() {
            bail!("{} is already served by another adapter", socket.display());
        }
        std::fs::remove_file(socket)
            .with_context(|| format!("remove stale {}", socket.display()))?;
    }
    let listener =
        UnixListener::bind(socket).with_context(|| format!("bind {}", socket.display()))?;
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

/// Only the user that runs the server may connect.
fn admit(stream: &UnixStream) -> Result<()> {
    let (mut uid, mut gid) = (0 as libc::uid_t, 0 as libc::gid_t);
    // SAFETY: valid fd and out-pointers.
    if unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } != 0 {
        bail!("getpeereid failed: {}", std::io::Error::last_os_error());
    }
    // SAFETY: geteuid has no preconditions.
    let me = unsafe { libc::geteuid() };
    if uid != me {
        bail!("peer uid {uid} is not the server's uid {me}");
    }
    Ok(())
}
