// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared host segments for `cuMemAllocHost` / `cuMemHostAlloc`.
//!
//! The server creates an anonymous POSIX shared-memory object, maps it, and
//! hands the descriptor to the client with `SCM_RIGHTS`. A copy between such a
//! segment and the device then carries only `{handle, offset, len}` on the
//! socket: the bytes are read from, or written to, the server's own mapping.
//!
//! The name is unlinked as soon as the object exists, so the memory lives
//! exactly as long as the two mappings and cannot be opened by anyone else.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{cu, CuError, CuResult};

/// Segment sizes are multiples of this. It is the arm64 page size, and a
/// multiple of the 4 KiB page an x86_64 process sees under Rosetta, so both
/// kinds of client can map a whole segment.
pub const SEGMENT_ALIGN: usize = 16 * 1024;
/// Largest single segment.
pub const MAX_SEGMENT_BYTES: usize = 768 * 1024 * 1024;
/// Segments one client may hold at once.
pub const MAX_SEGMENTS: usize = 64;

pub struct Segment {
    base: *mut u8,
    len: usize,
    fd: OwnedFd,
}

// SAFETY: the mapping is owned by this value and only touched through raw
// copies made while the owning session handles a request.
unsafe impl Send for Segment {}

impl Segment {
    pub fn create(bytes: usize) -> CuResult<Self> {
        if bytes == 0 || bytes > MAX_SEGMENT_BYTES {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!("host segment of {bytes} bytes is outside 1..={MAX_SEGMENT_BYTES}"),
            ));
        }
        let len = bytes.div_ceil(SEGMENT_ALIGN) * SEGMENT_ALIGN;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        // macOS limits shm names to 31 bytes.
        let name = CString::new(format!(
            "/cuad_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .expect("no interior NUL");
        let os_error = |what: &str| {
            CuError::new(
                cu::OUT_OF_MEMORY,
                format!("{what} failed: {}", std::io::Error::last_os_error()),
            )
        };
        // SAFETY: `name` is a valid C string; the mode is passed as the variadic argument.
        let raw = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600 as libc::c_uint,
            )
        };
        if raw < 0 {
            return Err(os_error("shm_open"));
        }
        // SAFETY: `raw` is a descriptor we just opened and own.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        // Unlink first: from here on only descriptor holders can reach the object.
        // SAFETY: valid C string.
        unsafe { libc::shm_unlink(name.as_ptr()) };
        // SAFETY: valid descriptor.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(os_error("ftruncate"));
        }
        // SAFETY: mapping a descriptor we own; the result is checked.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(os_error("mmap"));
        }
        Ok(Self {
            base: base as *mut u8,
            len,
            fd,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Pointer to `[offset, offset + len)` inside the segment, or an error if
    /// the range leaves it. The client also writes this memory, so callers
    /// copy through the raw pointer and never form a Rust reference to it.
    pub fn range(&self, offset: u64, len: u64) -> CuResult<*mut u8> {
        let end = offset.checked_add(len);
        if end.is_none_or(|end| end > self.len as u64) {
            return Err(CuError::new(
                cu::INVALID_VALUE,
                format!(
                    "range {offset}+{len} leaves the {}-byte host segment",
                    self.len
                ),
            ));
        }
        // SAFETY: `offset` is within the mapping (checked above).
        Ok(unsafe { self.base.add(offset as usize) })
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        // SAFETY: `base`/`len` describe the mapping created in `create`.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.len) };
    }
}

/// A connection that can carry a descriptor next to its byte stream.
pub trait FdChannel: std::io::Read + std::io::Write {
    /// Send one marker byte with `fd` attached (`SCM_RIGHTS`).
    fn send_fd(&mut self, fd: RawFd) -> std::io::Result<()>;
}

/// Marker byte that accompanies a descriptor.
pub const FD_MARKER: u8 = 0xFD;

impl FdChannel for UnixStream {
    fn send_fd(&mut self, fd: RawFd) -> std::io::Result<()> {
        let mut marker = [FD_MARKER];
        let mut iov = libc::iovec {
            iov_base: marker.as_mut_ptr() as *mut libc::c_void,
            iov_len: 1,
        };
        // SAFETY: CMSG_SPACE is a pure size computation.
        let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
        // u64 storage keeps the control buffer aligned for cmsghdr.
        let mut control = vec![0u64; space.div_ceil(8)];
        // SAFETY: zeroed msghdr is a valid starting value; every pointer set
        // below refers to a live local buffer of the stated length.
        let sent = unsafe {
            let mut msg: libc::msghdr = std::mem::zeroed();
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = space as libc::socklen_t;
            let header = libc::CMSG_FIRSTHDR(&msg);
            (*header).cmsg_level = libc::SOL_SOCKET;
            (*header).cmsg_type = libc::SCM_RIGHTS;
            (*header).cmsg_len =
                libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as libc::socklen_t;
            std::ptr::write_unaligned(libc::CMSG_DATA(header) as *mut RawFd, fd);
            loop {
                let n = libc::sendmsg(self.as_raw_fd(), &msg, 0);
                if n < 0
                    && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
                {
                    continue;
                }
                break n;
            }
        };
        if sent == 1 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_is_rounded_bounded_and_zero_filled() {
        let segment = Segment::create(100).unwrap();
        assert_eq!(segment.len(), SEGMENT_ALIGN);
        let p = segment.range(0, 100).unwrap();
        // SAFETY: the range was just validated; nobody else maps this segment.
        assert!(unsafe { std::slice::from_raw_parts(p, 100) }
            .iter()
            .all(|b| *b == 0));
        assert!(segment.range(SEGMENT_ALIGN as u64 - 4, 4).is_ok());
        assert_eq!(
            segment.range(SEGMENT_ALIGN as u64 - 4, 5).unwrap_err().code,
            cu::INVALID_VALUE
        );
        assert_eq!(
            segment.range(u64::MAX, 2).unwrap_err().code,
            cu::INVALID_VALUE
        );
        assert_eq!(Segment::create(0).err().unwrap().code, cu::INVALID_VALUE);
        assert_eq!(
            Segment::create(MAX_SEGMENT_BYTES + 1).err().unwrap().code,
            cu::INVALID_VALUE
        );
    }

    #[test]
    fn a_passed_descriptor_maps_the_same_memory() {
        let segment = Segment::create(SEGMENT_ALIGN).unwrap();
        let (mut tx, rx) = UnixStream::pair().unwrap();
        tx.send_fd(segment.fd()).unwrap();

        // Receive it the way the C forwarder does.
        let mut marker = [0u8; 1];
        let mut iov = libc::iovec {
            iov_base: marker.as_mut_ptr() as *mut libc::c_void,
            iov_len: 1,
        };
        let mut control = [0u64; 8];
        // SAFETY: all pointers refer to live locals of the stated sizes.
        let received: RawFd = unsafe {
            let mut msg: libc::msghdr = std::mem::zeroed();
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = std::mem::size_of_val(&control) as libc::socklen_t;
            assert_eq!(libc::recvmsg(rx.as_raw_fd(), &mut msg, 0), 1);
            let header = libc::CMSG_FIRSTHDR(&msg);
            assert!(!header.is_null());
            assert_eq!((*header).cmsg_type, libc::SCM_RIGHTS);
            std::ptr::read_unaligned(libc::CMSG_DATA(header) as *const RawFd)
        };
        assert_eq!(marker[0], FD_MARKER);
        // SAFETY: mapping a descriptor we received; checked below.
        let other = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SEGMENT_ALIGN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                received,
                0,
            )
        };
        assert_ne!(other, libc::MAP_FAILED);
        // SAFETY: both mappings cover SEGMENT_ALIGN bytes of the same object.
        unsafe {
            *(other as *mut u8).add(7) = 0x5a;
            assert_eq!(*segment.range(7, 1).unwrap(), 0x5a);
            libc::munmap(other, SEGMENT_ALIGN);
            libc::close(received);
        }
    }
}
