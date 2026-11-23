// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: glibc 2.17 imports of NVIDIA's Linux-aarch64
// `tileiras`, implemented on libSystem.
//
// Rules of this file:
//  * an import is bound only if its Darwin behaviour was reviewed for ABI
//    (argument passing, struct layout, constants). Everything else is left
//    unbound and the loader routes it to a trap that prints the symbol name and
//    exits 70 - it never "succeeds";
//  * errno: Linux code reads its own per-thread errno cell (Linux numbering).
//    Every `Thunk` binding is called through a machine-code thunk that moves a
//    Darwin errno into that cell after the call (see a64::errno_thunk). `Raw`
//    bindings are used for functions that read stack-passed variadics, never
//    return, or return twice;
//  * the locale model is "C" only, with glibc's `struct __locale_struct`
//    layout because the statically linked libstdc++ reads its ctype tables
//    directly;
//  * signal dispositions requested by the guest are stored (and read back by
//    later queries) but never installed in the kernel:
//    a Linux handler cannot receive a Darwin siginfo/ucontext. Faults are
//    reported by the helper's own handler instead.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU64, Ordering};
use std::sync::Mutex;

use binary_ko_loader::inproc::glibc_shim;

use crate::image::Binding;

type Ptr = *mut c_void;

extern "C" {
    fn dlsym(handle: Ptr, name: *const c_char) -> Ptr;
    fn __error() -> *mut c_int;
    fn _exit(code: c_int) -> !;
    fn abort() -> !;
    fn fflush(f: Ptr) -> c_int;
    fn calloc(n: usize, size: usize) -> Ptr;
    fn free(p: Ptr);
    fn write(fd: c_int, buf: *const u8, n: usize) -> isize;
    fn pthread_getspecific(k: u64) -> Ptr;
    fn pthread_setspecific(k: u64, v: Ptr) -> c_int;
    fn pthread_create(t: *mut Ptr, attr: Ptr, f: extern "C" fn(Ptr) -> Ptr, arg: Ptr) -> c_int;
    fn pthread_rwlock_init(l: Ptr, a: Ptr) -> c_int;
    fn pthread_rwlock_rdlock(l: Ptr) -> c_int;
    fn pthread_rwlock_wrlock(l: Ptr) -> c_int;
    fn pthread_rwlock_unlock(l: Ptr) -> c_int;
    fn pthread_self() -> Ptr;
    fn pthread_setname_np(name: *const c_char) -> c_int;
    fn sched_yield() -> c_int;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    fn ioctl(fd: c_int, req: u64, ...) -> c_int;
    fn fstat(fd: c_int, buf: *mut u8) -> c_int;
    fn stat(path: *const c_char, buf: *mut u8) -> c_int;
    fn lstat(path: *const c_char, buf: *mut u8) -> c_int;
    fn mmap(a: Ptr, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> Ptr;
    fn sysconf(name: c_int) -> i64;
    fn clock_gettime(id: c_int, ts: Ptr) -> c_int;
    fn getrlimit(res: c_int, lim: *mut u64) -> c_int;
    fn setrlimit(res: c_int, lim: *const u64) -> c_int;
    fn getrusage(who: c_int, out: *mut i64) -> c_int;
    fn uname(buf: *mut u8) -> c_int;
    fn strerror(e: c_int) -> *mut c_char;
    fn strsignal(s: c_int) -> *mut c_char;
    fn strlen(s: *const c_char) -> usize;
    fn strcmp(a: *const c_char, b: *const c_char) -> c_int;
    fn strtod(s: *const c_char, end: *mut *mut c_char) -> f64;
    fn strtof(s: *const c_char, end: *mut *mut c_char) -> f32;
    fn strftime(s: *mut c_char, max: usize, fmt: *const c_char, tm: Ptr) -> usize;
    fn raise(sig: c_int) -> c_int;
    fn arc4random_buf(buf: *mut u8, n: usize);
    fn getuid() -> u32;
    fn geteuid() -> u32;
    fn getgid() -> u32;
    fn getegid() -> u32;
}

const RTLD_DEFAULT: Ptr = -2isize as Ptr;

// ---------------------------------------------------------------------------
// process state shared with the loader
// ---------------------------------------------------------------------------
pub static TP_KEY: AtomicU64 = AtomicU64::new(0);
pub static X18_KEY: AtomicU64 = AtomicU64::new(0);
pub static TLS_BYTES: AtomicU64 = AtomicU64::new(0);
pub static SLIDE: AtomicU64 = AtomicU64::new(0);
pub static PHDR: AtomicU64 = AtomicU64::new(0);
pub static PHNUM: AtomicU64 = AtomicU64::new(0);
pub static ENTRY: AtomicU64 = AtomicU64::new(0);
/// Loaded tileiras image `[GUEST_LO, GUEST_HI)` and the helper's island /
/// thunk region `[GUEST_HI, ISLAND_END)` behind it (set by main after loading).
pub static GUEST_LO: AtomicU64 = AtomicU64::new(0);
pub static GUEST_HI: AtomicU64 = AtomicU64::new(0);
pub static ISLAND_END: AtomicU64 = AtomicU64::new(0);
pub static IMPORT_NAMES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static ENVIRON: AtomicPtr<*mut c_char> = AtomicPtr::new(std::ptr::null_mut());
static FALLBACK_ERRNO: AtomicI32 = AtomicI32::new(0);
static AT_RANDOM: Mutex<[u8; 16]> = Mutex::new([0; 16]);
static LINUX_STDIN: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static LINUX_STDOUT: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static LINUX_STDERR: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static STACK_CHK_GUARD: AtomicU64 = AtomicU64::new(0);

fn say(text: &str) {
    unsafe { write(2, text.as_ptr(), text.len()) };
}
fn fatal(text: &str) -> ! {
    say(&format!("tileiras-native-helper: FATAL: {text}\n"));
    unsafe { _exit(70) }
}

/// Linux AArch64 TLS variant I: TP -> 16-byte TCB, the PT_TLS block follows at
/// TP + 16 (p_align 8). The Linux errno cell is placed after the block.
fn errno_offset() -> usize {
    16 + ((TLS_BYTES.load(Ordering::Relaxed) as usize + 15) & !15)
}

/// Give the calling thread its Linux TLS block. Idempotent.
pub unsafe fn install_thread_block() -> Ptr {
    let key = TP_KEY.load(Ordering::Relaxed);
    let cur = pthread_getspecific(key);
    if !cur.is_null() {
        return cur;
    }
    let block = calloc(1, errno_offset() + 16);
    if block.is_null() || pthread_setspecific(key, block) != 0 {
        fatal("cannot allocate the Linux thread block");
    }
    pthread_setspecific(X18_KEY.load(Ordering::Relaxed), std::ptr::null_mut());
    block
}

unsafe fn linux_errno() -> *mut c_int {
    let tp = pthread_getspecific(TP_KEY.load(Ordering::Relaxed));
    if tp.is_null() {
        FALLBACK_ERRNO.as_ptr()
    } else {
        (tp as *mut u8).add(errno_offset()) as *mut c_int
    }
}

/// Darwin errno -> Linux errno (asm-generic numbering).
pub fn errno_d2l(e: c_int) -> c_int {
    match e {
        0..=10 | 12..=34 => e,
        11 => 35,
        35 => 11,
        36 => 115,
        37 => 114,
        38..=44 => e + 50,
        45 | 102 => 95,
        46..=58 => e + 50,
        60 => 110,
        61 => 111,
        62 => 40,
        63 => 36,
        64 => 112,
        65 => 113,
        66 => 39,
        68 => 87,
        69 => 122,
        70 => 116,
        71 => 66,
        77 => 37,
        78 => 38,
        84 => 75,
        89 => 125,
        90 => 43,
        91 => 42,
        92 => 84,
        94 => 74,
        95 => 72,
        96 => 61,
        97 => 67,
        98 => 63,
        99 => 60,
        100 => 71,
        101 => 62,
        104 => 131,
        105 => 130,
        _ => 5, // EIO: no Linux equivalent
    }
}
pub fn errno_l2d(e: c_int) -> c_int {
    (0..=105)
        .find(|d| errno_d2l(*d) == e && (*d == e || errno_d2l(*d) != 5))
        .unwrap_or(5)
}
/// Linux signal number -> Darwin (0 = no equivalent).
fn signal_l2d(s: c_int) -> c_int {
    match s {
        1..=6 | 8 | 9 | 11 | 13..=15 => s,
        7 => 10,
        10 => 30,
        12 => 31,
        17 => 20,
        18 => 19,
        19 => 17,
        20 => 18,
        21 | 22 => s,
        23 => 16,
        24..=28 => s,
        29 => 23,
        31 => 12,
        _ => 0,
    }
}

/// Called by every errno thunk after the Darwin function returned.
pub unsafe extern "C" fn errno_sync() {
    let d = __error();
    if *d != 0 {
        *linux_errno() = errno_d2l(*d);
        *d = 0;
    }
}
unsafe fn fail(darwin_errno: c_int) -> c_int {
    *__error() = darwin_errno;
    -1
}

/// getenv over the guest's own environment block (the helper passes an
/// explicit environment; libSystem's getenv would read the host's instead).
unsafe extern "C" fn sh_getenv(name: *const c_char) -> *const c_char {
    let mut entry = ENVIRON.load(Ordering::Acquire) as *const *const c_char;
    let n = strlen(name);
    if entry.is_null() || n == 0 {
        return std::ptr::null();
    }
    while !(*entry).is_null() {
        let e = *entry;
        if std::slice::from_raw_parts(e as *const u8, strlen(e))
            .starts_with(std::slice::from_raw_parts(name as *const u8, n))
            && *e.add(n) == b'=' as c_char
        {
            return e.add(n + 1);
        }
        entry = entry.add(1);
    }
    std::ptr::null()
}

/// The guest calls this whenever it touches errno (`errno = 0; f(); if (errno)`),
/// so a Darwin errno left behind by a `Raw` import is dropped here instead of
/// being attributed to the next thunked call.
unsafe extern "C" fn sh_errno_location() -> *mut c_int {
    *__error() = 0;
    linux_errno()
}
pub unsafe fn clear_host_errno() {
    *__error() = 0;
}
pub unsafe extern "C" fn keep_tp(value: Ptr) {
    pthread_setspecific(TP_KEY.load(Ordering::Relaxed), value);
}
pub unsafe extern "C" fn keep_x18(value: Ptr) {
    pthread_setspecific(X18_KEY.load(Ordering::Relaxed), value);
}

/// pthread functions report failure through their return value: translate it.
macro_rules! errno_returning {
    ($wrapper:ident, $cell:ident, $($arg:ident),*) => {
        static $cell: AtomicU64 = AtomicU64::new(0);
        unsafe extern "C" fn $wrapper($($arg: u64),*) -> c_int {
            let f: unsafe extern "C" fn($($arg: u64),*) -> c_int =
                std::mem::transmute($cell.load(Ordering::Relaxed) as usize);
            errno_d2l(f($($arg),*))
        }
    };
}
errno_returning!(w_join, T_JOIN, a, b);
errno_returning!(w_mutex_lock, T_MUTEX_LOCK, a);
errno_returning!(w_mutex_unlock, T_MUTEX_UNLOCK, a);
errno_returning!(w_cond_signal, T_COND_SIGNAL, a);
errno_returning!(w_cond_broadcast, T_COND_BROADCAST, a);
errno_returning!(w_cond_destroy, T_COND_DESTROY, a);

/// Reached through the per-import trap stub (x16 = import index).
pub unsafe extern "C" fn trap_unimplemented(index: u64) -> ! {
    let name = IMPORT_NAMES
        .lock()
        .ok()
        .and_then(|n| n.get(index as usize).cloned())
        .unwrap_or_else(|| "?".into());
    say(&format!(
        "tileiras-native-helper: unimplemented import '{name}' was called; no shim exists (exit 70)\n"
    ));
    _exit(70)
}

// ---------------------------------------------------------------------------
// guest stack roots and the C++ unwind gate
// ---------------------------------------------------------------------------
// `tileiras_guest_root(a0, a1, a2, f)` calls guest `f(a0, a1, a2)`. Its pc
// range marks the logical bottom of a guest stack (below main, a thread start
// routine, an exit handler), where "no unwind info" is the normal end of stack
// that glibc's __libc_start_main / start_thread give on Linux.
std::arch::global_asm!(
    ".text",
    ".p2align 2",
    ".globl _tileiras_guest_root",
    ".globl _tileiras_guest_root_end",
    "_tileiras_guest_root:",
    "stp x29, x30, [sp, #-16]!",
    "mov x29, sp",
    "blr x3",
    "ldp x29, x30, [sp], #16",
    "ret",
    "_tileiras_guest_root_end:",
);
extern "C" {
    fn tileiras_guest_root(a0: u64, a1: u64, a2: u64, f: u64) -> u64;
    fn tileiras_guest_root_end();
}

/// Called by the dl_iterate_phdr shim when no object claimed the caller's
/// query. If the query is the libgcc unwinder looking for a pc inside one of
/// the helper's own frames, a C++ exception is trying to cross a frame that
/// has no CFI: stop with a named diagnostic instead of the anonymous
/// std::terminate that would follow.
/// (symbol, image) when `pc` lies in the __TEXT segment of a native Mach-O
/// image (the helper or a libSystem dylib). mach_vm_region cannot answer this
/// for the dyld shared cache (it reports the whole submap as r--), so the
/// image's own load commands are read instead.
unsafe fn native_text_symbol(pc: u64) -> Option<(String, String)> {
    extern "C" {
        fn dladdr(addr: Ptr, info: *mut [usize; 4]) -> c_int;
    }
    let mut info = [0usize; 4]; // dli_fname, dli_fbase, dli_sname, dli_saddr
    if dladdr(pc as Ptr, &mut info) == 0 || info[1] == 0 {
        return None;
    }
    let header = info[1] as *const u32;
    if *header != 0xfeed_facf {
        return None;
    }
    let (commands, mut at) = (*header.add(4), (header as *const u8).add(32));
    for _ in 0..commands.min(4096) {
        let (cmd, size) = (*(at as *const u32), *(at.add(4) as *const u32));
        if cmd == 0x19 && std::slice::from_raw_parts(at.add(8), 7) == b"__TEXT\0" {
            let text_bytes = *(at.add(32) as *const u64);
            if pc - (info[1] as u64) >= text_bytes {
                return None;
            }
            let name = |p: usize| {
                if p != 0 {
                    cstr(p as *const c_char)
                } else {
                    "?".into()
                }
            };
            return Some((name(info[2]), name(info[0])));
        }
        if size < 8 {
            return None;
        }
        at = at.add(size as usize);
    }
    None
}

unsafe fn unwind_gate(data: Ptr) {
    use tileiras_native::unwind_gate::{classify, Layout, UnwindPc};
    // `data` is opaque to dl_iterate_phdr: only read it if 8 bytes are mapped
    // and readable there. A caller other than the unwinder is passed through
    // unless its first word happens to be an address inside helper / native
    // code (not the case for the pinned images' other uses; see README).
    if data.is_null() || (data as usize) % 8 != 0 {
        return;
    }
    match crate::image::region_protection(data as u64) {
        Some((_, end, prot)) if prot & 1 != 0 && end - data as u64 >= 8 => {}
        _ => return,
    }
    let pc = *(data as *const u64); // unw_eh_callback_data.pc
    let (lo, hi, island_end) = (
        GUEST_LO.load(Ordering::Relaxed),
        GUEST_HI.load(Ordering::Relaxed),
        ISLAND_END.load(Ordering::Relaxed),
    );
    let (mut guest, mut islands) = (vec![(lo, hi)], vec![(hi, island_end)]);
    if let Some(d) = loaded_dso() {
        guest.push((d.base, d.image_end));
        islands.push((d.image_end, d.end));
    }
    let root = tileiras_guest_root as *const () as u64;
    let layout = Layout {
        guest: &guest,
        islands: &islands,
        stack_root: (root, tileiras_guest_root_end as *const () as u64),
    };
    let native = native_text_symbol(pc);
    let what = match classify(pc, &layout, |_| native.is_some()) {
        UnwindPc::HelperIsland => "an import thunk / instruction island of the helper".to_string(),
        UnwindPc::NativeFrame => {
            let (symbol, image) = native.unwrap_or_default();
            format!("native function {symbol} in {image}")
        }
        _ => return,
    };
    fflush(std::ptr::null_mut());
    say(&format!(
        "tileiras-native-helper: C++ exception cannot unwind through a non-guest frame: pc {pc:#x} is {what}, which has no ELF unwind info. On Linux the exception would have propagated to the caller; this port stops here (exit 72)\n"
    ));
    _exit(72)
}

// ---------------------------------------------------------------------------
// startup / exit
// ---------------------------------------------------------------------------
static ATEXIT: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static CTORS_RAN: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" fn sh_cxa_atexit(f: Ptr, arg: Ptr, _dso: Ptr) -> c_int {
    ATEXIT.lock().unwrap().push((f as usize, arg as usize));
    0
}

unsafe extern "C" fn sh_exit(code: c_int) -> ! {
    loop {
        let next = ATEXIT.lock().unwrap().pop();
        match next {
            Some((f, a)) => {
                tileiras_guest_root(a as u64, 0, 0, f as u64);
            }
            None => break,
        }
    }
    fflush(std::ptr::null_mut());
    _exit(code)
}

/// glibc 2.17 `__libc_start_main(main, argc, argv, init, fini, rtld_fini, stack_end)`.
/// `init` is the binary's own `__libc_csu_init`, which walks .init_array once.
unsafe extern "C" fn sh_libc_start_main(
    main: u64,
    argc: c_int,
    argv: *mut *mut c_char,
    init: u64,
    fini: Ptr,
    _rtld_fini: Ptr,
    _stack_end: Ptr,
) -> ! {
    if CTORS_RAN.swap(1, Ordering::SeqCst) != 0 {
        fatal("__libc_start_main entered twice; constructors must run exactly once");
    }
    let envp = argv.add(argc as usize + 1);
    ENVIRON.store(envp, Ordering::SeqCst);
    *MAIN_ARGS.lock().unwrap() = (argc, argv as usize, envp as usize);
    if !fini.is_null() {
        ATEXIT.lock().unwrap().push((fini as usize, 0));
    }
    if init != 0 {
        tileiras_guest_root(argc as u64, argv as u64, envp as u64, init);
    }
    if std::env::var_os("TILEIRAS_NATIVE_STOP_AFTER_CTORS").is_some() {
        say("tileiras-native-helper: constructors completed; stopping before main as requested\n");
        _exit(0);
    }
    sh_exit(tileiras_guest_root(argc as u64, argv as u64, envp as u64, main) as c_int)
}

unsafe extern "C" fn sh_getauxval(kind: u64) -> u64 {
    match kind {
        3 => PHDR.load(Ordering::Relaxed),  // AT_PHDR
        4 => 56,                            // AT_PHENT
        5 => PHNUM.load(Ordering::Relaxed), // AT_PHNUM
        6 => 16384,                         // AT_PAGESZ (the real host page size)
        7 => 0,                             // AT_BASE: no interpreter is mapped
        9 => ENTRY.load(Ordering::Relaxed), // AT_ENTRY
        11 => getuid() as u64,
        12 => geteuid() as u64,
        13 => getgid() as u64,
        14 => getegid() as u64,
        16 => 0x3, // AT_HWCAP: FP | ASIMD only (no optional feature is promised)
        17 => 100, // AT_CLKTCK
        23 => 0,   // AT_SECURE
        25 => AT_RANDOM.lock().unwrap().as_ptr() as u64,
        26 => 0, // AT_HWCAP2
        _ => {
            *__error() = 2; // ENOENT, as glibc does for an absent entry
            0
        }
    }
}
pub fn auxv() -> Vec<(u64, u64)> {
    unsafe {
        arc4random_buf(AT_RANDOM.lock().unwrap().as_mut_ptr(), 16);
        [3u64, 4, 5, 6, 7, 9, 11, 12, 13, 14, 16, 17, 23, 25, 26]
            .iter()
            .map(|k| (*k, sh_getauxval(*k)))
            .collect()
    }
}

/// `struct dl_phdr_info` for the one loaded object; `dlpi_addr` is the ELF load
/// bias, which is exactly the slide, so libgcc's unwinder finds .eh_frame_hdr.
pub(crate) unsafe extern "C" fn sh_dl_iterate_phdr(
    callback: extern "C" fn(*mut u64, usize, Ptr) -> c_int,
    data: Ptr,
) -> c_int {
    // (pub(crate) for the self-check; the guest reaches it through its thunk.)
    let tp = pthread_getspecific(TP_KEY.load(Ordering::Relaxed)) as u64;
    let tls = |offset: u64| if tp == 0 { 0 } else { tp + offset };
    let objects = loaded_dso().map_or(1, |_| 2);
    let mut info = [
        SLIDE.load(Ordering::Relaxed),
        c"".as_ptr() as u64,
        PHDR.load(Ordering::Relaxed),
        PHNUM.load(Ordering::Relaxed), // Elf64_Half + padding
        objects,                       // dlpi_adds
        0,                             // dlpi_subs
        1,                             // dlpi_tls_modid
        tls(16),
    ];
    let r = callback(info.as_mut_ptr(), 64, data);
    if r != 0 {
        return r;
    }
    if let Some(d) = loaded_dso() {
        let offset = DSO_CONFIG
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |c| c.tls_offset);
        let mut info = [
            d.base,
            c"libnvvm.so.4".as_ptr() as u64,
            d.phdr,
            d.phnum,
            objects,
            0,
            2,
            tls(offset),
        ];
        let r = callback(info.as_mut_ptr(), 64, data);
        if r != 0 {
            return r;
        }
    }
    unwind_gate(data);
    0
}
/// `--unwind-gate-check`: drive the shim the way libgcc's _Unwind_Find_FDE
/// does, with a callback that claims nothing. Returns for pcs that are the
/// unwinder's own business; never returns (exit 72) for a helper frame.
pub unsafe fn unwind_gate_probe(pc: u64) {
    extern "C" fn claims_nothing(_info: *mut u64, _size: usize, _data: Ptr) -> c_int {
        0
    }
    let mut query = [pc, 0, 0, 0, 0, 0];
    sh_dl_iterate_phdr(claims_nothing, query.as_mut_ptr() as Ptr);
}
pub fn guest_root_pc() -> u64 {
    tileiras_guest_root as *const () as u64 + 12 // the return address of its `blr`
}

// ---------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------
/// Linux aarch64 `struct stat` (128 B) <- Darwin arm64 `struct stat` (144 B).
unsafe fn stat_out(d: &[u8; 144], out: *mut u8) {
    let rd = |o: usize, n: usize| -> u64 {
        let mut v = [0u8; 8];
        v[..n].copy_from_slice(&d[o..o + n]);
        u64::from_le_bytes(v)
    };
    let o = std::slice::from_raw_parts_mut(out, 128);
    o.fill(0);
    let mut put =
        |off: usize, n: usize, v: u64| o[off..off + n].copy_from_slice(&v.to_le_bytes()[..n]);
    put(0, 8, rd(0, 4)); // st_dev
    put(8, 8, rd(8, 8)); // st_ino
    put(16, 4, rd(4, 2)); // st_mode (S_IF* values are identical)
    put(20, 4, rd(6, 2)); // st_nlink
    put(24, 4, rd(16, 4)); // st_uid
    put(28, 4, rd(20, 4)); // st_gid
    put(32, 8, rd(24, 4)); // st_rdev
    put(48, 8, rd(96, 8)); // st_size
    put(56, 4, rd(112, 4)); // st_blksize
    put(64, 8, rd(104, 8)); // st_blocks
    for k in 0..6 {
        put(72 + 8 * k, 8, rd(32 + 8 * k, 8)); // st_atim, st_mtim, st_ctim
    }
}
unsafe extern "C" fn sh_fxstat(_ver: c_int, fd: c_int, out: *mut u8) -> c_int {
    let mut d = [0u8; 144];
    let r = fstat(fd, d.as_mut_ptr());
    if r == 0 {
        stat_out(&d, out);
    }
    r
}
unsafe extern "C" fn sh_xstat(_ver: c_int, path: *const c_char, out: *mut u8) -> c_int {
    let mut d = [0u8; 144];
    let r = stat(path, d.as_mut_ptr());
    if r == 0 {
        stat_out(&d, out);
    }
    r
}
unsafe extern "C" fn sh_lxstat(_ver: c_int, path: *const c_char, out: *mut u8) -> c_int {
    let mut d = [0u8; 144];
    let r = lstat(path, d.as_mut_ptr());
    if r == 0 {
        stat_out(&d, out);
    }
    r
}

/// Linux aarch64 open(2) flags -> Darwin. Unknown bits are refused.
fn open_flags_l2d(f: c_int) -> Option<c_int> {
    const MAP: [(c_int, c_int); 10] = [
        (0o100, 0x200),          // O_CREAT
        (0o200, 0x800),          // O_EXCL
        (0o400, 0x20000),        // O_NOCTTY
        (0o1000, 0x400),         // O_TRUNC
        (0o2000, 0x8),           // O_APPEND
        (0o4000, 0x4),           // O_NONBLOCK
        (0o40000, 0x10_0000),    // O_DIRECTORY (aarch64 value)
        (0o100000, 0x100),       // O_NOFOLLOW (aarch64 value)
        (0o2000000, 0x100_0000), // O_CLOEXEC
        (0o400000, 0),           // O_LARGEFILE (aarch64 value): implicit
    ];
    let mut out = f & 3;
    let mut rest = f & !3;
    for (l, d) in MAP {
        if rest & l != 0 {
            out |= d;
            rest &= !l;
        }
    }
    (rest == 0).then_some(out)
}
/// `open(path, flags, mode)`: Linux passes the variadic mode in x2.
unsafe extern "C" fn sh_open(path: *const c_char, flags: c_int, mode: c_int) -> c_int {
    if trace_on() {
        say(&format!("[trace] open({}, {flags:#o})\n", cstr(path)));
    }
    match open_flags_l2d(flags) {
        Some(f) => open(path, f, mode & 0o7777),
        None => {
            say(&format!(
                "tileiras-native-helper: open(): untranslated Linux flags {flags:#o}\n"
            ));
            fail(22)
        }
    }
}
/// fcntl: only the descriptor-flag commands, whose numbers and values agree.
unsafe extern "C" fn sh_fcntl(fd: c_int, cmd: c_int, arg: c_int) -> c_int {
    match cmd {
        1 => fcntl(fd, 1),      // F_GETFD
        2 => fcntl(fd, 2, arg), // F_SETFD (FD_CLOEXEC = 1 on both)
        _ => fatal(&format!("fcntl command {cmd} has no reviewed translation")),
    }
}
/// ioctl: only TIOCGWINSZ (same `struct winsize`).
unsafe extern "C" fn sh_ioctl(fd: c_int, request: u64, arg: Ptr) -> c_int {
    match request {
        0x5413 => ioctl(fd, 0x4008_7468, arg),
        _ => fatal(&format!(
            "ioctl request {request:#x} has no reviewed translation"
        )),
    }
}
unsafe extern "C" fn sh_mmap(
    a: Ptr,
    len: usize,
    prot: c_int,
    flags: c_int,
    fd: c_int,
    off: i64,
) -> Ptr {
    // MAP_SHARED 1, MAP_PRIVATE 2, MAP_FIXED 0x10 agree; MAP_ANONYMOUS 0x20 -> 0x1000;
    // MAP_NORESERVE 0x4000 -> 0x40.
    let known = 0x1 | 0x2 | 0x20 | 0x4000;
    if flags & !known != 0 || prot & !3 != 0 {
        // MAP_FIXED could overwrite the helper's own mappings; PROT_EXEC is
        // refused (no guest JIT is supported).
        say(&format!(
            "tileiras-native-helper: mmap(prot={prot:#x}, flags={flags:#x}) refused\n"
        ));
        *__error() = 22;
        return -1isize as Ptr;
    }
    let mut d = flags & 3;
    if flags & 0x20 != 0 {
        d |= 0x1000;
    }
    if flags & 0x4000 != 0 {
        d |= 0x40;
    }
    mmap(
        a,
        len,
        prot,
        d,
        if flags & 0x20 != 0 { -1 } else { fd },
        off,
    )
}

// ---------------------------------------------------------------------------
// system information
// ---------------------------------------------------------------------------
unsafe extern "C" fn sh_sysconf(name: c_int) -> i64 {
    let d = match name {
        0 => 1,    // _SC_ARG_MAX
        2 => 3,    // _SC_CLK_TCK
        4 => 5,    // _SC_OPEN_MAX
        30 => 29,  // _SC_PAGESIZE
        83 => 57,  // _SC_NPROCESSORS_CONF
        84 => 58,  // _SC_NPROCESSORS_ONLN
        85 => 200, // _SC_PHYS_PAGES
        _ => {
            *__error() = 22;
            return -1;
        }
    };
    sysconf(d)
}
unsafe extern "C" fn sh_get_nprocs() -> c_int {
    sysconf(58).max(1) as c_int
}
unsafe extern "C" fn sh_clock_gettime(id: c_int, ts: Ptr) -> c_int {
    let d = match id {
        0 => 0,  // CLOCK_REALTIME
        1 => 6,  // CLOCK_MONOTONIC
        2 => 12, // CLOCK_PROCESS_CPUTIME_ID
        3 => 16, // CLOCK_THREAD_CPUTIME_ID
        4 => 4,  // CLOCK_MONOTONIC_RAW
        _ => return fail(22),
    };
    clock_gettime(d, ts)
}
fn rlimit_l2d(resource: c_int) -> Option<c_int> {
    Some(match resource {
        0..=4 => resource, // CPU FSIZE DATA STACK CORE
        5 => 5,            // RSS
        6 => 7,            // NPROC
        7 => 8,            // NOFILE
        8 => 6,            // MEMLOCK
        9 => 5,            // AS
        _ => return None,
    })
}
unsafe extern "C" fn sh_getrlimit(resource: c_int, out: *mut u64) -> c_int {
    let Some(d) = rlimit_l2d(resource) else {
        return fail(22);
    };
    let r = getrlimit(d, out);
    if r == 0 {
        for k in 0..2 {
            if *out.add(k) >= 0x7fff_ffff_ffff_ffff {
                *out.add(k) = u64::MAX; // RLIM_INFINITY
            }
        }
    }
    r
}
unsafe extern "C" fn sh_setrlimit(resource: c_int, lim: *const u64) -> c_int {
    let Some(d) = rlimit_l2d(resource) else {
        return fail(22);
    };
    let v = [
        (*lim).min(0x7fff_ffff_ffff_ffff),
        (*lim.add(1)).min(0x7fff_ffff_ffff_ffff),
    ];
    setrlimit(d, v.as_ptr())
}
unsafe extern "C" fn sh_getrusage(who: c_int, out: *mut i64) -> c_int {
    let d = match who {
        0 => 0,
        -1 => -1,
        _ => return fail(22),
    };
    // Both layouts are 18 longs; Darwin's tv_usec is 32-bit plus padding the
    // kernel need not write, hence the zeroed native buffer and the masks.
    let mut native = [0i64; 18];
    let r = getrusage(d, native.as_mut_ptr());
    if r == 0 {
        native[1] &= 0xffff_ffff;
        native[3] &= 0xffff_ffff;
        native[4] /= 1024; // ru_maxrss: bytes -> kilobytes
        std::ptr::copy_nonoverlapping(native.as_ptr(), out, 18);
    }
    r
}
/// Linux `struct utsname` is 6 x 65 bytes; Darwin's is 5 x 256.
unsafe extern "C" fn sh_uname(out: *mut u8) -> c_int {
    let mut d = [0u8; 1280];
    let r = uname(d.as_mut_ptr());
    if r == 0 {
        std::ptr::write_bytes(out, 0, 390);
        for field in 0..5 {
            let src = &d[256 * field..256 * field + 256];
            let n = src.iter().position(|b| *b == 0).unwrap_or(256).min(64);
            std::ptr::copy_nonoverlapping(src.as_ptr(), out.add(65 * field), n);
        }
    }
    r
}
unsafe extern "C" fn sh_strerror(e: c_int) -> *mut c_char {
    strerror(errno_l2d(e))
}
/// GNU strerror_r: returns the message pointer.
unsafe extern "C" fn sh_strerror_r(e: c_int, buf: *mut c_char, n: usize) -> *mut c_char {
    let s = strerror(errno_l2d(e));
    if n > 0 {
        let len = strlen(s).min(n - 1);
        std::ptr::copy_nonoverlapping(s, buf, len);
        *buf.add(len) = 0;
    }
    buf
}
unsafe extern "C" fn sh_strsignal(s: c_int) -> *mut c_char {
    strsignal(signal_l2d(s))
}

// ---------------------------------------------------------------------------
// signals (recorded, not installed - see the header of this file)
// ---------------------------------------------------------------------------
/// Linux AArch64 `struct sigaction` is 152 bytes, handler first. SIG_DFL and
/// SIG_IGN are installed in the kernel (translated signal number) unless the
/// signal belongs to the helper's fault reporter or its alarm. A real guest
/// handler is stored and read back but NOT installed: it would receive Darwin's
/// siginfo/ucontext. The helper says so once on stderr.
static SIGACTIONS: Mutex<[[u8; 152]; 65]> = Mutex::new([[0; 152]; 65]);
static HANDLER_NOTE: AtomicI32 = AtomicI32::new(0);
unsafe extern "C" fn sh_sigaction(sig: c_int, new: *const u8, old: *mut u8) -> c_int {
    extern "C" {
        fn signal(sig: c_int, handler: usize) -> usize;
    }
    if !(1..=64).contains(&sig) || sig == 9 || sig == 19 {
        return fail(22);
    }
    let mut table = SIGACTIONS.lock().unwrap();
    let slot = &mut table[sig as usize];
    if !old.is_null() {
        std::ptr::copy_nonoverlapping(slot.as_ptr(), old, 152);
    }
    if !new.is_null() {
        std::ptr::copy_nonoverlapping(new, slot.as_mut_ptr(), 152);
        let handler = (new as *const u64).read_unaligned();
        let d = signal_l2d(sig);
        let reserved = matches!(d, 0 | 4 | 5 | 8 | 10 | 11 | 14);
        if handler <= 1 {
            if !reserved {
                signal(d, handler as usize);
            }
        } else if HANDLER_NOTE.swap(1, Ordering::Relaxed) == 0
            && std::env::var_os("TILEIRAS_NATIVE_TRACE").is_some()
        {
            say("tileiras-native-helper: note: guest signal handlers are stored but not installed\n");
        }
    }
    0
}
/// Linux `stack_t` = { sp, int flags, size }.
unsafe extern "C" fn sh_sigaltstack(_new: Ptr, old: *mut u64) -> c_int {
    if !old.is_null() {
        *old = 0;
        *old.add(1) = 2; // SS_DISABLE
        *old.add(2) = 0;
    }
    0
}
unsafe extern "C" fn sh_sigemptyset(set: *mut u8) -> c_int {
    std::ptr::write_bytes(set, 0, 128);
    0
}
unsafe extern "C" fn sh_sigfillset(set: *mut u8) -> c_int {
    std::ptr::write_bytes(set, 0xff, 128);
    0
}
unsafe extern "C" fn sh_sigaddset(set: *mut u8, sig: c_int) -> c_int {
    if !(1..=64).contains(&sig) {
        return fail(22);
    }
    *set.add((sig as usize - 1) / 8) |= 1 << ((sig - 1) % 8);
    0
}
static SIGMASK: Mutex<[u8; 128]> = Mutex::new([0; 128]);
/// The guest's mask is kept as data: no guest handler is ever installed, so
/// blocking has no observable effect inside the guest.
unsafe extern "C" fn sh_sigprocmask(how: c_int, new: *const u8, old: *mut u8) -> c_int {
    let mut mask = SIGMASK.lock().unwrap();
    if !old.is_null() {
        std::ptr::copy_nonoverlapping(mask.as_ptr(), old, 128);
    }
    if !new.is_null() {
        for k in 0..128 {
            let n = *new.add(k);
            mask[k] = match how {
                0 => mask[k] | n,
                1 => mask[k] & !n,
                2 => n,
                _ => return fail(22),
            };
        }
    }
    0
}
unsafe extern "C" fn sh_pthread_sigmask(how: c_int, new: *const u8, old: *mut u8) -> c_int {
    if sh_sigprocmask(how, new, old) == 0 {
        0
    } else {
        *__error() = 0;
        22
    }
}
unsafe extern "C" fn sh_raise(sig: c_int) -> c_int {
    match signal_l2d(sig) {
        0 => fail(22),
        d => {
            fflush(std::ptr::null_mut());
            raise(d)
        }
    }
}

// ---------------------------------------------------------------------------
// threads
// ---------------------------------------------------------------------------
struct Start {
    f: extern "C" fn(Ptr) -> Ptr,
    arg: Ptr,
}
extern "C" fn thread_entry(p: Ptr) -> Ptr {
    unsafe {
        let s = Box::from_raw(p as *mut Start);
        // The block is deliberately never freed (816 bytes per thread): guest
        // pthread-key destructors run after this function returns and may
        // still read TLS, and the order of key destructors is unspecified.
        install_thread_block();
        tileiras_guest_root(s.arg as u64, 0, 0, s.f as usize as u64) as Ptr
    }
}
unsafe extern "C" fn sh_pthread_create(
    t: *mut Ptr,
    attr: Ptr,
    f: extern "C" fn(Ptr) -> Ptr,
    arg: Ptr,
) -> c_int {
    let s = Box::into_raw(Box::new(Start { f, arg }));
    let r = pthread_create(t, attr, thread_entry, s as Ptr);
    if r != 0 {
        drop(Box::from_raw(s));
    }
    errno_d2l(r)
}
/// Linux `pthread_once_t` is one int: 0 = new, 1 = running, 2 = done.
unsafe extern "C" fn sh_pthread_once(once: *const AtomicI32, f: extern "C" fn()) -> c_int {
    loop {
        match (*once).compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                f();
                (*once).store(2, Ordering::Release);
                return 0;
            }
            Err(2) => return 0,
            Err(_) => {
                sched_yield();
            }
        }
    }
}
/// Linux `pthread_rwlock_t` is 56 bytes and zero when statically initialised;
/// a Darwin rwlock (200 bytes) is allocated on first use and its address kept
/// in glibc's `__pad1` (offset 32, zero in every static initialiser; `__flags`
/// at 48 is not, e.g. PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP).
unsafe fn rwlock(l: *mut u8) -> Ptr {
    let slot = &*(l.add(32) as *const AtomicPtr<c_void>);
    let cur = slot.load(Ordering::Acquire);
    if !cur.is_null() {
        return cur;
    }
    let n = calloc(1, 256);
    pthread_rwlock_init(n, std::ptr::null_mut());
    match slot.compare_exchange(std::ptr::null_mut(), n, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => n,
        Err(w) => {
            free(n);
            w
        }
    }
}
unsafe extern "C" fn sh_rwlock_init(l: *mut u8, _attr: Ptr) -> c_int {
    std::ptr::write_bytes(l, 0, 56);
    0
}
unsafe extern "C" fn sh_rwlock_destroy(l: *mut u8) -> c_int {
    extern "C" {
        fn pthread_rwlock_destroy(l: Ptr) -> c_int;
    }
    let slot = &*(l.add(32) as *const AtomicPtr<c_void>);
    let n = slot.swap(std::ptr::null_mut(), Ordering::AcqRel);
    if !n.is_null() {
        pthread_rwlock_destroy(n);
        free(n);
    }
    0
}
/// `__cxa_finalize(dso)` runs when a shared object is unloaded; objects are
/// never unloaded here (dlclose keeps them mapped), so there is nothing to run.
unsafe extern "C" fn sh_cxa_finalize(_dso: Ptr) {}
unsafe extern "C" fn sh_rwlock_rdlock(l: *mut u8) -> c_int {
    errno_d2l(pthread_rwlock_rdlock(rwlock(l)))
}
unsafe extern "C" fn sh_rwlock_wrlock(l: *mut u8) -> c_int {
    errno_d2l(pthread_rwlock_wrlock(rwlock(l)))
}
unsafe extern "C" fn sh_rwlock_unlock(l: *mut u8) -> c_int {
    errno_d2l(pthread_rwlock_unlock(rwlock(l)))
}
/// Linux: (thread, name); Darwin can only name the calling thread.
unsafe extern "C" fn sh_pthread_setname_np(thread: Ptr, name: *const c_char) -> c_int {
    if thread == pthread_self() {
        errno_d2l(pthread_setname_np(name))
    } else {
        34 // ERANGE is the documented failure; naming another thread is unsupported
    }
}
unsafe extern "C" fn sh_sched_getaffinity(_pid: c_int, size: usize, set: *mut u8) -> c_int {
    let n = (sysconf(58).max(1) as usize).min(size * 8);
    std::ptr::write_bytes(set, 0, size);
    for cpu in 0..n {
        *set.add(cpu / 8) |= 1 << (cpu % 8);
    }
    0
}
unsafe extern "C" fn sh_sched_cpucount(size: usize, set: *const u8) -> c_int {
    (0..size).map(|k| (*set.add(k)).count_ones() as c_int).sum()
}
unsafe extern "C" fn sh_sched_cpualloc(count: usize) -> Ptr {
    calloc(1, count.div_ceil(64) * 8)
}

/// `syscall(number, ...)`: Linux passes the variadic arguments in x1..x6. Only
/// numbers with a reviewed Darwin equivalent are served (AArch64 numbering).
unsafe extern "C" fn sh_syscall(number: i64, a1: u64, _a2: u64, _a3: u64) -> i64 {
    extern "C" {
        fn pthread_threadid_np(thread: Ptr, id: *mut u64) -> c_int;
        fn getpid() -> c_int;
    }
    match number {
        178 => {
            // gettid: a process-unique thread id
            let mut id = 0u64;
            pthread_threadid_np(std::ptr::null_mut(), &mut id);
            id as i64
        }
        172 => getpid() as i64,
        _ => fatal(&format!(
            "syscall({number}, {a1:#x}, ...) has no reviewed translation"
        )),
    }
}

// ---------------------------------------------------------------------------
// locale: "C" only, glibc `struct __locale_struct` layout
// ---------------------------------------------------------------------------
#[repr(C)]
struct LinuxLocale {
    locales: [u64; 13],
    ctype_b: *const u16,
    ctype_tolower: *const i32,
    ctype_toupper: *const i32,
    names: [*const c_char; 13],
}
static C_LOCALE: AtomicPtr<LinuxLocale> = AtomicPtr::new(std::ptr::null_mut());

fn glibc_class(c: u8) -> u16 {
    let (up, lo, dig) = (
        c.is_ascii_uppercase(),
        c.is_ascii_lowercase(),
        c.is_ascii_digit(),
    );
    let mut v = 0u16;
    let mut set = |on: bool, bit: u16| {
        if on {
            v |= bit
        }
    };
    set(up, 0x100);
    set(lo, 0x200);
    set(up || lo, 0x400);
    set(dig, 0x800);
    set(c.is_ascii_hexdigit(), 0x1000);
    set(matches!(c, b' ' | 9..=13), 0x2000);
    set((0x20..0x7f).contains(&c), 0x4000);
    set((0x21..0x7f).contains(&c), 0x8000);
    set(c == b' ' || c == b'\t', 0x1);
    set(c < 0x20 || c == 0x7f, 0x2);
    set(c.is_ascii_punctuation(), 0x4);
    set(up || lo || dig, 0x8);
    v
}
unsafe fn c_locale() -> *mut LinuxLocale {
    let cur = C_LOCALE.load(Ordering::Acquire);
    if !cur.is_null() {
        return cur;
    }
    // Tables are indexed -128..255 like glibc's.
    let b: &'static mut [u16; 384] = Box::leak(Box::new([0; 384]));
    let lower: &'static mut [i32; 384] = Box::leak(Box::new([0; 384]));
    let upper: &'static mut [i32; 384] = Box::leak(Box::new([0; 384]));
    for i in 0..384usize {
        let c = i as i32 - 128;
        lower[i] = c;
        upper[i] = c;
        if (0..128).contains(&c) {
            b[i] = glibc_class(c as u8);
            lower[i] = (c as u8).to_ascii_lowercase() as i32;
            upper[i] = (c as u8).to_ascii_uppercase() as i32;
        }
    }
    let l = Box::into_raw(Box::new(LinuxLocale {
        locales: [0; 13],
        ctype_b: b.as_ptr().add(128),
        ctype_tolower: lower.as_ptr().add(128),
        ctype_toupper: upper.as_ptr().add(128),
        names: [c"C".as_ptr(); 13],
    }));
    match C_LOCALE.compare_exchange(std::ptr::null_mut(), l, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => l,
        Err(w) => w,
    }
}
unsafe extern "C" fn sh_newlocale(
    _mask: c_int,
    name: *const c_char,
    _base: Ptr,
) -> *mut LinuxLocale {
    let n = CStr::from_ptr(name).to_bytes();
    if n.is_empty() || n == b"C" || n == b"POSIX" {
        c_locale()
    } else {
        *__error() = 2; // ENOENT: only the C locale exists here
        std::ptr::null_mut()
    }
}
unsafe extern "C" fn sh_duplocale(l: *mut LinuxLocale) -> *mut LinuxLocale {
    if l as isize == -1 {
        c_locale()
    } else {
        l
    }
}
unsafe extern "C" fn sh_freelocale(_l: *mut LinuxLocale) {}
/// Returns the previous locale; with one locale that is always the C locale
/// (LC_GLOBAL_LOCALE, -1, when the thread never set one, as glibc reports).
unsafe extern "C" fn sh_uselocale(_l: *mut LinuxLocale) -> *mut LinuxLocale {
    -1isize as *mut LinuxLocale
}
unsafe extern "C" fn sh_nl_langinfo_l(item: c_int, _l: Ptr) -> *const c_char {
    match item {
        14 => c"ANSI_X3.4-1968".as_ptr(), // CODESET
        0x10000 => c".".as_ptr(),         // RADIXCHAR
        _ => c"".as_ptr(),                // glibc returns "" for the C locale / unknown items
    }
}
unsafe extern "C" fn sh_ctype_get_mb_cur_max() -> usize {
    1
}
unsafe extern "C" fn sh_strcoll_l(a: *const c_char, b: *const c_char, _l: Ptr) -> c_int {
    strcmp(a, b)
}
unsafe extern "C" fn sh_strxfrm_l(
    dst: *mut c_char,
    src: *const c_char,
    n: usize,
    _l: Ptr,
) -> usize {
    let len = strlen(src);
    if n > 0 {
        let k = len.min(n - 1);
        std::ptr::copy_nonoverlapping(src, dst, k);
        *dst.add(k) = 0;
    }
    len
}
unsafe extern "C" fn sh_strtod_l(s: *const c_char, end: *mut *mut c_char, _l: Ptr) -> f64 {
    strtod(s, end) // the helper never calls setlocale: the host locale is "C"
}
unsafe extern "C" fn sh_strtof_l(s: *const c_char, end: *mut *mut c_char, _l: Ptr) -> f32 {
    strtof(s, end)
}
unsafe extern "C" fn sh_strftime_l(
    s: *mut c_char,
    max: usize,
    fmt: *const c_char,
    tm: Ptr,
    _l: Ptr,
) -> usize {
    strftime(s, max, fmt, tm)
}
unsafe extern "C" fn sh_towlower_l(c: u32, _l: Ptr) -> u32 {
    if c < 128 {
        (c as u8).to_ascii_lowercase() as u32
    } else {
        c
    }
}
unsafe extern "C" fn sh_towupper_l(c: u32, _l: Ptr) -> u32 {
    if c < 128 {
        (c as u8).to_ascii_uppercase() as u32
    } else {
        c
    }
}
const WCTYPES: [(&[u8], u16); 12] = [
    (b"upper", 0x100),
    (b"lower", 0x200),
    (b"alpha", 0x400),
    (b"digit", 0x800),
    (b"xdigit", 0x1000),
    (b"space", 0x2000),
    (b"print", 0x4000),
    (b"graph", 0x8000),
    (b"blank", 0x1),
    (b"cntrl", 0x2),
    (b"punct", 0x4),
    (b"alnum", 0x8),
];
unsafe extern "C" fn sh_wctype_l(name: *const c_char, _l: Ptr) -> u64 {
    let n = CStr::from_ptr(name).to_bytes();
    WCTYPES
        .iter()
        .find(|(k, _)| *k == n)
        .map_or(0, |(_, bit)| *bit as u64)
}
unsafe extern "C" fn sh_iswctype_l(c: u32, kind: u64, _l: Ptr) -> c_int {
    (c < 128 && glibc_class(c as u8) as u64 & kind != 0) as c_int
}
/// No message catalogues exist: gettext's documented result is the msgid.
unsafe extern "C" fn sh_dgettext(_domain: *const c_char, msgid: *const c_char) -> *const c_char {
    msgid
}
// C-locale multibyte conversions on Linux's 8-byte mbstate_t (never touched:
// the C locale is stateless). Darwin's mbstate_t is 128 bytes, so libSystem's
// functions must not be handed the guest's object.
unsafe extern "C" fn sh_btowc(c: c_int) -> u32 {
    if (0..128).contains(&c) {
        c as u32
    } else {
        u32::MAX
    }
}
unsafe extern "C" fn sh_wctob(c: u32) -> c_int {
    if c < 128 {
        c as c_int
    } else {
        -1
    }
}
unsafe extern "C" fn sh_mbrtowc(out: *mut u32, s: *const u8, n: usize, _st: Ptr) -> usize {
    if s.is_null() {
        return 0;
    }
    if n == 0 {
        return usize::MAX - 1;
    }
    if *s >= 128 {
        *__error() = 92; // EILSEQ
        return usize::MAX;
    }
    if !out.is_null() {
        *out = *s as u32;
    }
    (*s != 0) as usize
}
unsafe extern "C" fn sh_wcrtomb(out: *mut u8, c: u32, _st: Ptr) -> usize {
    if out.is_null() {
        return 1;
    }
    if c >= 128 {
        *__error() = 92;
        return usize::MAX;
    }
    *out = c as u8;
    1
}

// ---------------------------------------------------------------------------
// dynamic loading: there is no Linux ld.so. Exactly one object can be opened,
// the pinned libNVVM; any other name or handle is a named trap (exit 70).
// ---------------------------------------------------------------------------
pub fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TILEIRAS_NATIVE_TRACE").is_some())
}
unsafe fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        "(null)".into()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
/// Where the one admissible shared object comes from (set by main).
pub struct DsoConfig {
    pub path: std::path::PathBuf,
    pub manifest: std::path::PathBuf,
    pub tp_slot: u64,
    pub x18_slot: u64,
    pub tls_offset: u64,
    pub tls_capacity: u64,
}
pub const LIBNVVM_SHA256: &str = "f3cb2690f62c668924a373482a9bc748ace90b94bf13adce8e76a1f077682077";
pub const LIBNVVM_MANIFEST_SHA256: &str =
    "c3facbc0b7d78e729b9d8c259c2ef64beb5da7d3f9c22602f246c62268fd6ac0";
pub static DSO_CONFIG: Mutex<Option<DsoConfig>> = Mutex::new(None);
static DSO: AtomicPtr<crate::dso::Dso> = AtomicPtr::new(std::ptr::null_mut());
static DSO_LOCK: Mutex<()> = Mutex::new(());
thread_local! {
    /// Pending dlerror text and the string last handed to the guest, per
    /// thread (glibc's dlerror state is per thread too).
    static DL_ERROR: std::cell::RefCell<(Option<CString>, Option<CString>)> = const { std::cell::RefCell::new((None, None)) };
}
fn set_dl_error(text: String) {
    DL_ERROR.with(|e| e.borrow_mut().0 = CString::new(text).ok());
}
/// Thread that is running libNVVM's constructors (0: none).
static DSO_INIT_OWNER: AtomicU64 = AtomicU64::new(0);
pub static DSO_BASE: AtomicU64 = AtomicU64::new(0);
pub static DSO_END: AtomicU64 = AtomicU64::new(0);
static MAIN_ARGS: Mutex<(c_int, usize, usize)> = Mutex::new((0, 0, 0));

pub fn loaded_dso() -> Option<&'static crate::dso::Dso> {
    unsafe { DSO.load(Ordering::Acquire).as_ref() }
}

fn load_libnvvm() -> Result<&'static crate::dso::Dso, String> {
    use tileiras_native::{patches, sha256};
    let config = DSO_CONFIG.lock().unwrap();
    let c = config
        .as_ref()
        .ok_or("no libNVVM was configured for this run")?;
    let elf = std::fs::read(&c.path).map_err(|e| format!("{}: {e}", c.path.display()))?;
    if sha256::hex(&elf) != LIBNVVM_SHA256 {
        return Err(format!(
            "{} is not the admitted libnvvm.so.4.0.0 (13.2.51 linux-sbsa)",
            c.path.display()
        ));
    }
    let text = std::fs::read(&c.manifest).map_err(|e| format!("{}: {e}", c.manifest.display()))?;
    if sha256::hex(&text) != LIBNVVM_MANIFEST_SHA256 {
        return Err(format!(
            "{} is not the pinned libNVVM patch manifest",
            c.manifest.display()
        ));
    }
    let manifest = patches::parse(std::str::from_utf8(&text).map_err(|e| e.to_string())?)?;
    let trap_index_base = IMPORT_NAMES.lock().unwrap().len();
    let dso = crate::dso::load(&crate::dso::Inputs {
        elf: &elf,
        elf_sha256: LIBNVVM_SHA256,
        manifest: &manifest,
        tp_slot: c.tp_slot,
        x18_slot: c.x18_slot,
        tls_offset: c.tls_offset,
        tls_capacity: c.tls_capacity,
        resolve: &resolve,
        trap_handler: trap_unimplemented as *const () as usize as u64,
        errno_sync: errno_sync as *const () as usize as u64,
        trap_index_base,
    })?;
    IMPORT_NAMES
        .lock()
        .unwrap()
        .extend(dso.imports_trapping.iter().cloned());
    DSO_BASE.store(dso.base, Ordering::Release);
    DSO_END.store(dso.end, Ordering::Release);
    Ok(Box::leak(Box::new(dso)))
}

/// Names under which the pinned libNVVM may be requested (any directory).
pub fn admitted_library(requested: &str) -> bool {
    matches!(
        requested.rsplit('/').next(),
        Some("libnvvm.so" | "libnvvm.so.4" | "libnvvm.so.4.0.0")
    )
}
fn trap_dl(call: &str, what: &str) -> ! {
    say(&format!(
        "tileiras-native-helper: {call}({what}) is not admitted: only the pinned libnvvm.so can be loaded; no shim exists (exit 70)\n"
    ));
    unsafe { _exit(70) }
}
unsafe fn admitted_handle(call: &str, handle: Ptr) -> &'static crate::dso::Dso {
    match loaded_dso() {
        Some(d) if d.base == handle as u64 => d,
        _ => trap_dl(call, &format!("handle {handle:p}")),
    }
}

/// dlopen: exactly one object exists, NVIDIA's libNVVM, loaded from the
/// pinned file whatever directory the guest names. Any other name (including
/// NULL, the main program) is a named trap. Constructors run once, on the
/// first caller's thread, before any caller gets the handle.
pub unsafe extern "C" fn sh_dlopen(name: *const c_char, _flags: c_int) -> Ptr {
    let requested = cstr(name);
    if name.is_null() || !admitted_library(&requested) {
        trap_dl("dlopen", &requested);
    }
    // A constructor that re-opens libNVVM gets the handle of the object being
    // initialised, as ld.so does for a recursive dlopen.
    let me = pthread_self() as u64;
    if DSO_INIT_OWNER.load(Ordering::Acquire) == me {
        if let Some(d) = loaded_dso() {
            return d.base as Ptr;
        }
    }
    // Held across the constructors: a second thread blocks here until the
    // object is fully initialised (as on ld.so's load lock; a constructor that
    // waits for such a thread would deadlock there too - the alarm ends it).
    let _guard = DSO_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(d) = loaded_dso() {
        return d.base as Ptr;
    }
    let dso = match load_libnvvm() {
        Ok(d) => d,
        Err(e) => {
            say(&format!(
                "tileiras-native-helper: dlopen({requested}) failed: {e}\n"
            ));
            set_dl_error(e);
            return std::ptr::null_mut();
        }
    };
    DSO.store(dso as *const _ as *mut _, Ordering::Release);
    if trace_on() {
        say(&format!(
            "[trace] dlopen({requested}) -> libnvvm at {:#x}, {} constructors, relocations {:?}, {} unimplemented imports\n",
            dso.base, dso.init.len(), dso.relocations, dso.imports_trapping.len()
        ));
    }
    let (argc, argv, envp) = *MAIN_ARGS.lock().unwrap();
    DSO_INIT_OWNER.store(me, Ordering::Release);
    for f in &dso.init {
        let f: extern "C" fn(c_int, usize, usize) = std::mem::transmute(*f as usize);
        f(argc, argv, envp);
    }
    DSO_INIT_OWNER.store(0, Ordering::Release);
    dso.base as Ptr
}
/// dlsym on the libNVVM handle; a missing symbol is NULL + dlerror, as on
/// Linux (callers probe optional entry points). Any other handle traps.
pub unsafe extern "C" fn sh_dlsym(handle: Ptr, name: *const c_char) -> Ptr {
    let found = admitted_handle("dlsym", handle)
        .exports
        .get(&cstr(name))
        .copied();
    if trace_on() {
        say(&format!(
            "[trace] dlsym({}) -> {:#x}\n",
            cstr(name),
            found.unwrap_or(0)
        ));
    }
    match found {
        Some(a) => a as Ptr,
        None => {
            set_dl_error(format!("undefined symbol: {}", cstr(name)));
            std::ptr::null_mut()
        }
    }
}
/// The object stays mapped for the life of the process (its destructors run
/// from the exit handlers it registered). Any other handle traps.
unsafe extern "C" fn sh_dlclose(handle: Ptr) -> c_int {
    admitted_handle("dlclose", handle);
    0
}
unsafe extern "C" fn sh_dlerror() -> *const c_char {
    // The returned string stays valid until this thread's next dlerror call.
    DL_ERROR.with(|e| {
        let mut e = e.borrow_mut();
        e.1 = e.0.take();
        e.1.as_ref().map_or(std::ptr::null(), |s| s.as_ptr())
    })
}
unsafe extern "C" fn sh_dladdr(_addr: Ptr, _info: Ptr) -> c_int {
    0
}
unsafe extern "C" fn sh_stack_chk_fail() -> ! {
    say("tileiras-native-helper: __stack_chk_fail in guest code\n");
    abort()
}
unsafe extern "C" fn sh_aligned_alloc(align: usize, size: usize) -> Ptr {
    extern "C" {
        fn posix_memalign(out: *mut Ptr, align: usize, size: usize) -> c_int;
    }
    let mut p = std::ptr::null_mut();
    // glibc accepts any power-of-two alignment; Darwin's aligned_alloc is stricter.
    match posix_memalign(&mut p, align.max(8), size) {
        0 => p,
        e => {
            *__error() = e;
            std::ptr::null_mut()
        }
    }
}

/// `long double` is IEEE binary128 on Linux AArch64 (passed in q0) and plain
/// double on Darwin, so libSystem's floorl cannot be used. Bit-exact floor.
pub fn floor_binary128(bits: u128) -> u128 {
    const FRACTION: u32 = 112;
    let negative = bits >> 127 != 0;
    let exponent = ((bits >> FRACTION) & 0x7fff) as u32;
    if exponent < 16383 {
        // |x| < 1
        return if bits << 1 == 0 {
            bits
        } else if negative {
            (1u128 << 127) | (16383u128 << FRACTION) // -1.0
        } else {
            0
        };
    }
    if exponent >= 16383 + FRACTION {
        return bits; // integral, infinity or NaN
    }
    let mask = (1u128 << (FRACTION - (exponent - 16383))) - 1;
    if bits & mask == 0 {
        return bits;
    }
    (if negative { bits + mask } else { bits }) & !mask
}
// NEON vector arguments travel in v0 under AAPCS64, which is what is needed.
#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn sh_floorl(
    x: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::uint8x16_t {
    std::mem::transmute(floor_binary128(std::mem::transmute::<_, u128>(x)))
}

// ---------------------------------------------------------------------------
// child processes: only `ptxas`, replaced by the pinned native one
// ---------------------------------------------------------------------------
/// Script that prints the verified native ptxas (tools/ptxas-resolve.sh).
pub static PTXAS_RESOLVER: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

enum FileAction {
    Open {
        fd: c_int,
        path: CString,
        flags: c_int,
        mode: c_int,
    },
    Dup2 {
        fd: c_int,
        new_fd: c_int,
    },
}
/// glibc's `posix_spawn_file_actions_t` is 80 opaque bytes; the action list
/// lives on the host heap and its address is kept in the second word.
unsafe fn file_actions(fa: *mut u64) -> &'static mut Vec<FileAction> {
    let p = *fa.add(1) as *mut Vec<FileAction>;
    if p.is_null() || *fa != 0x7470_7366_6163_7473 {
        fatal("posix_spawn file actions used before posix_spawn_file_actions_init");
    }
    &mut *p
}
unsafe extern "C" fn sh_spawn_fa_init(fa: *mut u64) -> c_int {
    std::ptr::write_bytes(fa as *mut u8, 0, 80);
    *fa = 0x7470_7366_6163_7473;
    *fa.add(1) = Box::into_raw(Box::new(Vec::<FileAction>::new())) as u64;
    0
}
unsafe extern "C" fn sh_spawn_fa_destroy(fa: *mut u64) -> c_int {
    drop(Box::from_raw(file_actions(fa) as *mut Vec<FileAction>));
    std::ptr::write_bytes(fa as *mut u8, 0, 80);
    0
}
unsafe extern "C" fn sh_spawn_fa_addopen(
    fa: *mut u64,
    fd: c_int,
    path: *const c_char,
    flags: c_int,
    mode: c_int,
) -> c_int {
    let Some(flags) = open_flags_l2d(flags) else {
        return 22;
    };
    file_actions(fa).push(FileAction::Open {
        fd,
        path: CStr::from_ptr(path).to_owned(),
        flags,
        mode: mode & 0o7777,
    });
    0
}
unsafe extern "C" fn sh_spawn_fa_adddup2(fa: *mut u64, fd: c_int, new_fd: c_int) -> c_int {
    file_actions(fa).push(FileAction::Dup2 { fd, new_fd });
    0
}

/// The only program the guest may start, under any directory.
pub fn admitted_program(requested: &str) -> bool {
    requested.rsplit('/').next() == Some("ptxas")
}

/// Trust root for the child compiler, independent of anything the resolver
/// prints: `$HOME/.config/cuda-oxide/ptxas-native.pin`, written by
/// portkit/ptxas-native/build.sh after its verify step (solution 105).
const PTXAS_PIN: &str = ".config/cuda-oxide/ptxas-native.pin";

/// `binary_sha256=` of the pin file text (64 lowercase hex digits).
pub fn pinned_ptxas_sha256(pin_text: &str) -> Result<String, String> {
    let mut values = pin_text
        .lines()
        .filter_map(|l| l.strip_prefix("binary_sha256="));
    match (values.next(), values.next()) {
        (Some(v), None)
            if v.len() == 64
                && v.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
        {
            Ok(v.to_string())
        }
        _ => Err("the ptxas pin file has no single well-formed binary_sha256".into()),
    }
}

/// What may be executed as the guest's `ptxas`: the file the resolver selected
/// must be `<pinned sha256>/ptxas-darwin-arm64` (the resolver's private,
/// content-addressed copy) AND its bytes must hash to the PINNED digest. It is
/// re-read and re-hashed before EVERY spawn, so a resolver that answers with
/// any other program, or a file replaced after selection, is refused. (A
/// same-user race between this hash and exec remains, as the resolver
/// documents.)
pub fn pinned_compiler_ok(
    path: &std::path::Path,
    bytes: &[u8],
    pinned_sha256: &str,
) -> Result<(), String> {
    let dir = path
        .parent()
        .and_then(|d| d.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name != "ptxas-darwin-arm64" || dir != pinned_sha256 {
        return Err(format!(
            "{} is not the pinned compiler's content-addressed copy ({pinned_sha256}/ptxas-darwin-arm64)",
            path.display()
        ));
    }
    let have = tileiras_native::sha256::hex(bytes);
    if have != pinned_sha256 {
        return Err(format!(
            "{} has SHA-256 {have}, not the pinned {pinned_sha256}",
            path.display()
        ));
    }
    Ok(())
}

fn native_ptxas() -> Result<CString, String> {
    let path = resolved_ptxas()?;
    let file = std::path::PathBuf::from(path.to_string_lossy().into_owned());
    let home = std::env::var_os("HOME").ok_or("HOME missing")?;
    let pin_path = std::path::PathBuf::from(home).join(PTXAS_PIN);
    let pin =
        std::fs::read_to_string(&pin_path).map_err(|e| format!("{}: {e}", pin_path.display()))?;
    let bytes = std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    pinned_compiler_ok(&file, &bytes, &pinned_ptxas_sha256(&pin)?)?;
    Ok(path)
}

fn resolved_ptxas() -> Result<CString, String> {
    static PATH: Mutex<Option<CString>> = Mutex::new(None);
    let mut cached = PATH.lock().unwrap();
    if let Some(p) = cached.as_ref() {
        return Ok(p.clone());
    }
    let resolver = PTXAS_RESOLVER
        .lock()
        .unwrap()
        .clone()
        .ok_or("no --ptxas-resolver was given")?;
    let out = std::process::Command::new(&resolver)
        .arg("--print-path")
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("{}: {e}", resolver.display()))?;
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if !out.status.success() || !line.starts_with('/') || line.contains('\n') {
        return Err(format!(
            "{} --print-path selected no verified ptxas: {}",
            resolver.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let c = CString::new(line).map_err(|e| e.to_string())?;
    *cached = Some(c.clone());
    Ok(c)
}

/// posix_spawn: the only admissible program is `ptxas`, and what runs is the
/// native compiler selected by the resolver script, with the guest's argv[1..],
/// environment and file actions. Any other target is a named trap.
unsafe extern "C" fn sh_posix_spawn(
    pid: *mut c_int,
    path: *const c_char,
    fa: *mut u64,
    attr: Ptr,
    argv: *const *const c_char,
    envp: *const *const c_char,
) -> c_int {
    extern "C" {
        fn posix_spawn(
            pid: *mut c_int,
            path: *const c_char,
            fa: *const Ptr,
            attr: Ptr,
            argv: *const *const c_char,
            envp: *const *const c_char,
        ) -> c_int;
        fn posix_spawn_file_actions_init(fa: *mut Ptr) -> c_int;
        fn posix_spawn_file_actions_destroy(fa: *mut Ptr) -> c_int;
        fn posix_spawn_file_actions_addopen(
            fa: *mut Ptr,
            fd: c_int,
            path: *const c_char,
            flags: c_int,
            mode: u16,
        ) -> c_int;
        fn posix_spawn_file_actions_adddup2(fa: *mut Ptr, fd: c_int, new_fd: c_int) -> c_int;
    }
    let requested = cstr(path);
    if path.is_null() || !admitted_program(&requested) || argv.is_null() || (*argv).is_null() {
        say(&format!("tileiras-native-helper: spawn of '{requested}' is not admitted; only ptxas is (exit 70)\n"));
        _exit(70)
    }
    if !attr.is_null() {
        fatal("posix_spawn with spawn attributes has no reviewed translation");
    }
    let program = match native_ptxas() {
        Ok(p) => p,
        Err(e) => fatal(&format!("cannot run ptxas for the guest: {e}")),
    };
    // argv[1..] is forwarded unchanged except for one spelling: the native
    // wrapper takes `--opt-level=N`, tileiras writes `--opt-level N`.
    let mut owned: Vec<CString> = Vec::new();
    let mut k = 1;
    while !(*argv.add(k)).is_null() {
        let arg = CStr::from_ptr(*argv.add(k));
        let value = *argv.add(k + 1);
        if arg.to_bytes() == b"--opt-level" && !value.is_null() {
            let mut joined = b"--opt-level=".to_vec();
            joined.extend_from_slice(CStr::from_ptr(value).to_bytes());
            owned.push(CString::new(joined).unwrap());
            k += 2;
        } else {
            owned.push(arg.to_owned());
            k += 1;
        }
    }
    // Optional evidence: keep a copy of the PTX that libNVVM produced.
    if let Some(keep) = std::env::var_os("TILEIRAS_NATIVE_KEEP_PTX") {
        if let Some(ptx) = owned.iter().find(|a| a.to_bytes().ends_with(b".ptx")) {
            if std::fs::copy(ptx.to_string_lossy().as_ref(), &keep).is_err() {
                fatal("TILEIRAS_NATIVE_KEEP_PTX: cannot copy the PTX");
            }
        }
    }
    let mut args: Vec<*const c_char> = vec![program.as_ptr()];
    args.extend(owned.iter().map(|a| a.as_ptr()));
    args.push(std::ptr::null());
    if trace_on() {
        let shown: Vec<String> = args[..args.len() - 1].iter().map(|a| cstr(*a)).collect();
        say(&format!(
            "[trace] posix_spawn({requested}) -> native {shown:?}\n"
        ));
    }
    let mut native: Ptr = std::ptr::null_mut();
    let r = posix_spawn_file_actions_init(&mut native);
    if r != 0 {
        return errno_d2l(r);
    }
    if !fa.is_null() {
        for action in file_actions(fa).iter() {
            let r = match action {
                FileAction::Open {
                    fd,
                    path,
                    flags,
                    mode,
                } => posix_spawn_file_actions_addopen(
                    &mut native,
                    *fd,
                    path.as_ptr(),
                    *flags,
                    *mode as u16,
                ),
                FileAction::Dup2 { fd, new_fd } => {
                    posix_spawn_file_actions_adddup2(&mut native, *fd, *new_fd)
                }
            };
            if r != 0 {
                posix_spawn_file_actions_destroy(&mut native);
                return errno_d2l(r);
            }
        }
    }
    let mut child = 0;
    let r = posix_spawn(
        &mut child,
        program.as_ptr(),
        &native,
        std::ptr::null_mut(),
        args.as_ptr(),
        envp,
    );
    posix_spawn_file_actions_destroy(&mut native);
    if r == 0 && !pid.is_null() {
        *pid = child;
    }
    errno_d2l(r)
}
fn signal_d2l(signal: c_int) -> c_int {
    (1..=31).find(|l| signal_l2d(*l) == signal).unwrap_or(9)
}
/// Wait status: exit codes agree; the terminating signal (low 7 bits) or the
/// stop signal (second byte when the low byte is 0x7f) is translated. Darwin
/// reports a continued child as "stopped by SIGCONT"; Linux uses 0xffff.
fn wait_status_d2l(status: c_int) -> c_int {
    let signal = status & 0x7f;
    if signal == 0 {
        return status;
    }
    if signal == 0x7f {
        let stop = (status >> 8) & 0xff;
        return if stop == 19 {
            0xffff
        } else {
            (signal_d2l(stop) << 8) | 0x7f
        };
    }
    (status & !0x7f) | signal_d2l(signal)
}
unsafe extern "C" fn sh_wait4(
    pid: c_int,
    status: *mut c_int,
    options: c_int,
    usage: *mut i64,
) -> c_int {
    extern "C" {
        fn wait4(pid: c_int, status: *mut c_int, options: c_int, usage: *mut i64) -> c_int;
    }
    if options & !3 != 0 {
        return fail(22); // only WNOHANG | WUNTRACED agree between the systems
    }
    let (mut st, mut native) = (0, [0i64; 18]);
    let r = wait4(pid, &mut st, options, native.as_mut_ptr());
    if r > 0 {
        if !status.is_null() {
            *status = wait_status_d2l(st);
        }
        if !usage.is_null() {
            native[1] &= 0xffff_ffff;
            native[3] &= 0xffff_ffff;
            native[4] /= 1024;
            std::ptr::copy_nonoverlapping(native.as_ptr(), usage, 18);
        }
    }
    r
}
unsafe extern "C" fn sh_waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int {
    sh_wait4(pid, status, options, std::ptr::null_mut())
}
unsafe extern "C" fn sh_wait(status: *mut c_int) -> c_int {
    sh_wait4(-1, status, 0, std::ptr::null_mut())
}

// ---------------------------------------------------------------------------
// resolver
// ---------------------------------------------------------------------------
/// libSystem functions whose prototype, struct layouts and constants agree
/// with glibc for AArch64 (LP64, 32-bit wchar_t, 64-bit off_t, same `struct tm`,
/// `struct timespec`, `struct iovec`, opaque FILE*/DIR*/pthread_t).
const PASSTHROUGH: &[&str] = &[
    "feof",
    "fopen",
    "fseek",
    "ftell",
    "getpagesize",
    "mkdir",
    "strcpy",
    "strtof",
    "calloc",
    "free",
    "malloc",
    "realloc",
    "posix_memalign",
    "memchr",
    "memcmp",
    "memcpy",
    "memmove",
    "memset",
    "strchr",
    "strcmp",
    "strdup",
    "strlen",
    "strncmp",
    "strncpy",
    "strrchr",
    "strtod",
    "strtok_r",
    "strtol",
    "strtoll",
    "strtoull",
    "qsort",
    "rand",
    "srand",
    "tolower",
    "toupper",
    "isalpha",
    "isspace",
    "isalnum",
    "islower",
    "isupper",
    "isxdigit",
    "acos",
    "acosf",
    "acosh",
    "acoshf",
    "asin",
    "asinf",
    "asinh",
    "asinhf",
    "atan",
    "atan2",
    "atan2f",
    "atanf",
    "atanh",
    "atanhf",
    "ceil",
    "cos",
    "cosf",
    "cosh",
    "coshf",
    "erf",
    "erfc",
    "erfcf",
    "erff",
    "exp",
    "exp2",
    "exp2f",
    "expf",
    "expm1",
    "expm1f",
    "fabs",
    "floor",
    "fmod",
    "log",
    "log10",
    "log10f",
    "log1p",
    "log1pf",
    "log2",
    "log2f",
    "logb",
    "logf",
    "pow",
    "powf",
    "round",
    "sin",
    "sinf",
    "sinh",
    "sinhf",
    "sqrt",
    "sqrtf",
    "tan",
    "tanf",
    "tanh",
    "tanhf",
    "trunc",
    "wcscmp",
    "wcslen",
    "wmemchr",
    "wmemcpy",
    "wmemmove",
    "wmemset",
    "fclose",
    "fflush",
    "fileno",
    "fputc",
    "fputs",
    "fread",
    "fwrite",
    "getc",
    "putc",
    "putchar",
    "ungetc",
    "read",
    "write",
    "writev",
    "close",
    "dup",
    "dup2",
    "lseek",
    "pread",
    "unlink",
    "access",
    "chdir",
    "getcwd",
    "readlink",
    "realpath",
    "remove",
    "isatty",
    "getpid",
    "getuid",
    "opendir",
    "closedir",
    "time",
    "nanosleep",
    "localtime_r",
    "strftime",
    "munmap",
    "posix_madvise",
    "pthread_attr_destroy",
    "pthread_attr_init",
    "pthread_attr_setstacksize",
    "pthread_self",
    "abort",
    "_exit",
];
const ALIASES: &[(&str, &str)] = &[
    ("fopen64", "fopen"),
    ("fseeko64", "fseeko"),
    ("ftello64", "ftello"),
    ("lseek64", "lseek"),
    ("__sched_cpufree", "free"),
    ("_IO_getc", "getc"),
    ("_IO_putc", "putc"),
];
/// Shims shared with the native ptxas port (binary-ko-loader glibc_shim). The
/// variadic ones read the caller's stack and therefore cannot take the thunk.
const FROM_PTXAS_RAW: &[&str] = &["printf", "fprintf", "snprintf", "sprintf", "vsnprintf"];
/// Fixed-argument shims shared with the native ptxas port.
const FROM_PTXAS_THUNK: &[&str] = &[
    "pthread_key_create",
    "pthread_key_delete",
    "pthread_getspecific",
    "pthread_setspecific",
    "gettimeofday",
    "readdir",
    "pthread_cond_init",
    "pthread_mutex_destroy",
    "pthread_mutex_init",
    "pthread_mutexattr_destroy",
    "pthread_mutexattr_init",
    "pthread_mutexattr_settype",
    "sched_get_priority_max",
    "sched_get_priority_min",
    "sem_destroy",
    "sem_init",
    "sem_post",
    "vfprintf",
    "vsprintf",
];
const WEAK_NULL: &[&str] = &[
    "_ITM_deregisterTMCloneTable",
    "_ITM_registerTMCloneTable",
    "__gmon_start__",
];

fn native(name: &str) -> Option<u64> {
    let c = CString::new(name).ok()?;
    let p = unsafe { dlsym(RTLD_DEFAULT, c.as_ptr()) };
    (!p.is_null()).then_some(p as u64)
}

pub fn resolve(name: &str) -> Option<Binding> {
    macro_rules! thunk {
        ($f:expr) => {
            return Some(Binding::Thunk($f as usize as u64))
        };
    }
    macro_rules! raw {
        ($f:expr) => {
            return Some(Binding::Raw($f as usize as u64))
        };
    }
    unsafe {
        extern "C" {
            static __stdinp: Ptr;
            static __stdoutp: Ptr;
            static __stderrp: Ptr;
        }
        match name {
            "stdin" => {
                LINUX_STDIN.store(__stdinp, Ordering::Release);
                raw!(LINUX_STDIN.as_ptr())
            }
            "stdout" => {
                LINUX_STDOUT.store(__stdoutp, Ordering::Release);
                raw!(LINUX_STDOUT.as_ptr())
            }
            "stderr" => {
                LINUX_STDERR.store(__stderrp, Ordering::Release);
                raw!(LINUX_STDERR.as_ptr())
            }
            "environ" | "__environ" | "_environ" => raw!(ENVIRON.as_ptr()),
            "__stack_chk_guard" => {
                let mut g = [0u8; 8];
                arc4random_buf(g.as_mut_ptr(), 8);
                g[0] = 0;
                STACK_CHK_GUARD.store(u64::from_le_bytes(g), Ordering::Release);
                raw!(STACK_CHK_GUARD.as_ptr())
            }
            _ => {}
        }
    }
    match name {
        "__errno_location" => raw!(sh_errno_location as unsafe extern "C" fn() -> *mut c_int),
        "__libc_start_main" => raw!(sh_libc_start_main as *const c_void),
        "exit" => raw!(sh_exit as unsafe extern "C" fn(c_int) -> !),
        // Raw: the thunk preserves only the low halves of v0..v3.
        "floorl" => raw!(sh_floorl as *const c_void),
        "__stack_chk_fail" => raw!(sh_stack_chk_fail as unsafe extern "C" fn() -> !),
        "_setjmp" => return native("_setjmp").map(Binding::Raw),
        "longjmp" => return native("_longjmp").map(Binding::Raw),
        "__cxa_atexit" => thunk!(sh_cxa_atexit as unsafe extern "C" fn(Ptr, Ptr, Ptr) -> c_int),
        "getenv" => thunk!(sh_getenv as *const c_void),
        "getauxval" | "__getauxval" => thunk!(sh_getauxval as unsafe extern "C" fn(u64) -> u64),
        "dl_iterate_phdr" => thunk!(sh_dl_iterate_phdr as *const c_void),
        "__fxstat" | "__fxstat64" => thunk!(sh_fxstat as *const c_void),
        "__xstat" => thunk!(sh_xstat as *const c_void),
        "__lxstat" => thunk!(sh_lxstat as *const c_void),
        "open" => thunk!(sh_open as *const c_void),
        "fcntl" => thunk!(sh_fcntl as *const c_void),
        "ioctl" => thunk!(sh_ioctl as *const c_void),
        "mmap" => thunk!(sh_mmap as *const c_void),
        "sysconf" => thunk!(sh_sysconf as *const c_void),
        "get_nprocs" => thunk!(sh_get_nprocs as *const c_void),
        "clock_gettime" => thunk!(sh_clock_gettime as *const c_void),
        "getrlimit" => thunk!(sh_getrlimit as *const c_void),
        "setrlimit" => thunk!(sh_setrlimit as *const c_void),
        "getrusage" => thunk!(sh_getrusage as *const c_void),
        "uname" => thunk!(sh_uname as *const c_void),
        "strerror" => thunk!(sh_strerror as *const c_void),
        "strerror_r" => thunk!(sh_strerror_r as *const c_void),
        "strsignal" => thunk!(sh_strsignal as *const c_void),
        "sigaction" => thunk!(sh_sigaction as *const c_void),
        "sigaltstack" => thunk!(sh_sigaltstack as *const c_void),
        "sigemptyset" => thunk!(sh_sigemptyset as *const c_void),
        "sigfillset" => thunk!(sh_sigfillset as *const c_void),
        "sigaddset" => thunk!(sh_sigaddset as *const c_void),
        "sigprocmask" => thunk!(sh_sigprocmask as *const c_void),
        "pthread_sigmask" => thunk!(sh_pthread_sigmask as *const c_void),
        "raise" => thunk!(sh_raise as *const c_void),
        "pthread_create" => thunk!(sh_pthread_create as *const c_void),
        "pthread_once" => thunk!(sh_pthread_once as *const c_void),
        "__pthread_key_create" => {
            return glibc_shim::resolve("pthread_key_create").map(Binding::Raw)
        }
        "pthread_rwlock_init" => thunk!(sh_rwlock_init as *const c_void),
        "pthread_rwlock_destroy" => thunk!(sh_rwlock_destroy as *const c_void),
        "__cxa_finalize" => thunk!(sh_cxa_finalize as *const c_void),
        "pthread_rwlock_rdlock" => thunk!(sh_rwlock_rdlock as *const c_void),
        "pthread_rwlock_wrlock" => thunk!(sh_rwlock_wrlock as *const c_void),
        "pthread_rwlock_unlock" => thunk!(sh_rwlock_unlock as *const c_void),
        "pthread_setname_np" => thunk!(sh_pthread_setname_np as *const c_void),
        "sched_getaffinity" => thunk!(sh_sched_getaffinity as *const c_void),
        "__sched_cpucount" => thunk!(sh_sched_cpucount as *const c_void),
        "__sched_cpualloc" => thunk!(sh_sched_cpualloc as *const c_void),
        "__newlocale" => thunk!(sh_newlocale as *const c_void),
        "__duplocale" => thunk!(sh_duplocale as *const c_void),
        "__freelocale" => thunk!(sh_freelocale as *const c_void),
        "__uselocale" => thunk!(sh_uselocale as *const c_void),
        "__nl_langinfo_l" => thunk!(sh_nl_langinfo_l as *const c_void),
        "__ctype_get_mb_cur_max" => thunk!(sh_ctype_get_mb_cur_max as *const c_void),
        "__strcoll_l" => thunk!(sh_strcoll_l as *const c_void),
        "__strxfrm_l" => thunk!(sh_strxfrm_l as *const c_void),
        "__strtod_l" => thunk!(sh_strtod_l as *const c_void),
        "__strtof_l" => thunk!(sh_strtof_l as *const c_void),
        "__strftime_l" => thunk!(sh_strftime_l as *const c_void),
        "__towlower_l" => thunk!(sh_towlower_l as *const c_void),
        "__towupper_l" => thunk!(sh_towupper_l as *const c_void),
        "__wctype_l" => thunk!(sh_wctype_l as *const c_void),
        "__iswctype_l" => thunk!(sh_iswctype_l as *const c_void),
        "dgettext" => thunk!(sh_dgettext as *const c_void),
        "btowc" => thunk!(sh_btowc as *const c_void),
        "wctob" => thunk!(sh_wctob as *const c_void),
        "mbrtowc" => thunk!(sh_mbrtowc as *const c_void),
        "wcrtomb" => thunk!(sh_wcrtomb as *const c_void),
        "dlopen" => thunk!(sh_dlopen as *const c_void),
        "dlsym" => thunk!(sh_dlsym as *const c_void),
        "dlerror" => thunk!(sh_dlerror as *const c_void),
        "dladdr" => thunk!(sh_dladdr as *const c_void),
        "posix_spawn_file_actions_init" => thunk!(sh_spawn_fa_init as *const c_void),
        "posix_spawn_file_actions_destroy" => thunk!(sh_spawn_fa_destroy as *const c_void),
        "posix_spawn_file_actions_addopen" => thunk!(sh_spawn_fa_addopen as *const c_void),
        "posix_spawn_file_actions_adddup2" => thunk!(sh_spawn_fa_adddup2 as *const c_void),
        "posix_spawn" => thunk!(sh_posix_spawn as *const c_void),
        "wait4" => thunk!(sh_wait4 as *const c_void),
        "waitpid" => thunk!(sh_waitpid as *const c_void),
        "wait" => thunk!(sh_wait as *const c_void),
        "dlclose" => thunk!(sh_dlclose as *const c_void),
        "syscall" => thunk!(sh_syscall as *const c_void),
        "aligned_alloc" => thunk!(sh_aligned_alloc as *const c_void),
        _ => {}
    }
    if WEAK_NULL.contains(&name) {
        return Some(Binding::WeakNull);
    }
    let wrapped = |cell: &AtomicU64, target: Option<u64>, wrapper: u64| {
        target.map(|t| {
            cell.store(t, Ordering::Release);
            Binding::Thunk(wrapper)
        })
    };
    match name {
        "pthread_join" => return wrapped(&T_JOIN, native(name), w_join as *const c_void as u64),
        "pthread_mutex_lock" => {
            return wrapped(
                &T_MUTEX_LOCK,
                glibc_shim::resolve(name),
                w_mutex_lock as *const c_void as u64,
            )
        }
        "pthread_mutex_unlock" => {
            return wrapped(
                &T_MUTEX_UNLOCK,
                glibc_shim::resolve(name),
                w_mutex_unlock as *const c_void as u64,
            )
        }
        // The shared shim already returns a Linux error number.
        "pthread_cond_wait" => return glibc_shim::resolve(name).map(Binding::Thunk),
        "pthread_cond_signal" => {
            return wrapped(
                &T_COND_SIGNAL,
                glibc_shim::resolve(name),
                w_cond_signal as *const c_void as u64,
            )
        }
        "pthread_cond_broadcast" => {
            return wrapped(
                &T_COND_BROADCAST,
                glibc_shim::resolve(name),
                w_cond_broadcast as *const c_void as u64,
            )
        }
        "pthread_cond_destroy" => {
            return wrapped(
                &T_COND_DESTROY,
                glibc_shim::resolve(name),
                w_cond_destroy as *const c_void as u64,
            )
        }
        _ => {}
    }
    if FROM_PTXAS_RAW.contains(&name) {
        return glibc_shim::resolve(name).map(Binding::Raw);
    }
    if FROM_PTXAS_THUNK.contains(&name) {
        return glibc_shim::resolve(name).map(Binding::Thunk);
    }
    if let Some((_, target)) = ALIASES.iter().find(|(n, _)| *n == name) {
        return native(target).map(Binding::Thunk);
    }
    if PASSTHROUGH.contains(&name) {
        return native(name).map(if matches!(name, "abort" | "_exit") {
            Binding::Raw
        } else {
            Binding::Thunk
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_tables_are_consistent() {
        assert_eq!(errno_d2l(35), 11); // EAGAIN
        assert_eq!(errno_d2l(11), 35); // EDEADLK
        assert_eq!(errno_d2l(60), 110); // ETIMEDOUT
        assert_eq!(errno_d2l(63), 36); // ENAMETOOLONG
        assert_eq!(errno_d2l(2), 2);
        for l in [1, 2, 11, 13, 17, 22, 34, 35, 36, 38, 39, 40, 84, 95, 110] {
            assert_eq!(errno_d2l(errno_l2d(l)), l, "linux errno {l}");
        }
    }

    #[test]
    fn floor_binary128_matches_ieee() {
        let q = |sign: u128, exp: u128, frac: u128| (sign << 127) | (exp << 112) | frac;
        let one = q(0, 16383, 0);
        assert_eq!(floor_binary128(q(0, 16383, 1 << 111)), one); // 1.5 -> 1
        assert_eq!(floor_binary128(q(1, 16383, 1 << 111)), q(1, 16384, 0)); // -1.5 -> -2
        assert_eq!(floor_binary128(q(0, 16382, 0)), 0); // 0.5 -> 0
        assert_eq!(floor_binary128(q(1, 16382, 0)), q(1, 16383, 0)); // -0.5 -> -1
        assert_eq!(floor_binary128(q(1, 0, 0)), q(1, 0, 0)); // -0 -> -0
        assert_eq!(
            floor_binary128(q(0, 16384, 3 << 110)),
            q(0, 16384, 1 << 111)
        ); // 3.5 -> 3
        assert_eq!(floor_binary128(q(1, 16384, 3 << 110)), q(1, 16385, 0)); // -3.5 -> -4
        assert_eq!(floor_binary128(q(0, 0x7fff, 0)), q(0, 0x7fff, 0)); // inf
        assert_eq!(floor_binary128(q(0, 16383 + 112, 5)), q(0, 16383 + 112, 5));
    }

    #[test]
    fn only_libnvvm_and_ptxas_are_admitted() {
        for ok in [
            "libnvvm.so",
            "libnvvm.so.4",
            "/opt/cuda/nvvm/lib64/libnvvm.so",
            "nvvm/lib64/libnvvm.so.4.0.0",
        ] {
            assert!(admitted_library(ok), "{ok}");
        }
        for bad in [
            "",
            "(null)",
            "libc.so.6",
            "libnvvm.so.5",
            "libnvvm.so/x",
            "libnvidia-gpucomp.so.595.04",
            "xlibnvvm.so",
            "libnvvm.so ",
        ] {
            assert!(!admitted_library(bad), "{bad}");
        }
        for ok in ["ptxas", "/usr/local/cuda/bin/ptxas", "./ptxas"] {
            assert!(admitted_program(ok), "{ok}");
        }
        for bad in [
            "",
            "sh",
            "/bin/sh",
            "ptxas.sh",
            "ptxas/",
            "/tmp/ptxas/evil",
            "nvptxas",
            "PTXAS",
        ] {
            assert!(!admitted_program(bad), "{bad}");
        }
    }

    #[test]
    fn the_spawned_compiler_must_be_the_pinned_one() {
        use std::path::Path;
        let body = b"stand-in for the pinned compiler";
        let pin = tileiras_native::sha256::hex(body);
        let good = Path::new("/c").join(&pin).join("ptxas-darwin-arm64");
        assert!(pinned_compiler_ok(&good, body, &pin).is_ok());
        // Replaced content, even though the path is still the pinned one.
        assert!(pinned_compiler_ok(&good, b"replaced", &pin).is_err());
        // A self-consistent content address that is NOT the pinned digest.
        let other = b"some other program";
        let other_sha = tileiras_native::sha256::hex(other);
        assert!(pinned_compiler_ok(
            &Path::new("/c").join(&other_sha).join("ptxas-darwin-arm64"),
            other,
            &pin
        )
        .is_err());
        // Pinned content under another name or directory.
        assert!(pinned_compiler_ok(&Path::new("/c").join(&pin).join("sh"), body, &pin).is_err());
        assert!(
            pinned_compiler_ok(Path::new("/usr/local/bin/ptxas-darwin-arm64"), body, &pin).is_err()
        );
        assert!(pinned_compiler_ok(Path::new("/ptxas-darwin-arm64"), body, &pin).is_err());

        let text = format!("# comment\nbinary_sha256={pin}\narchive_sha256=00\n");
        assert_eq!(pinned_ptxas_sha256(&text).unwrap(), pin);
        assert!(pinned_ptxas_sha256("archive_sha256=00\n").is_err());
        assert!(
            pinned_ptxas_sha256(&format!("binary_sha256={pin}\nbinary_sha256={pin}\n")).is_err()
        );
        assert!(pinned_ptxas_sha256(&format!("binary_sha256={}\n", pin.to_uppercase())).is_err());
        assert!(pinned_ptxas_sha256("binary_sha256=abc\n").is_err());
    }

    #[test]
    fn wait_status_keeps_exit_codes_and_translates_signals() {
        assert_eq!(wait_status_d2l(0), 0);
        assert_eq!(wait_status_d2l(3 << 8), 3 << 8); // exit(3)
        assert_eq!(wait_status_d2l(9), 9); // SIGKILL
        assert_eq!(wait_status_d2l(10), 7); // Darwin SIGBUS -> Linux SIGBUS
        assert_eq!(wait_status_d2l(0x80 | 11), 0x80 | 11); // core flag kept
        assert_eq!(wait_status_d2l(0x117f), 0x137f); // stopped by SIGSTOP (17 -> 19)
        assert_eq!(wait_status_d2l(0x127f), 0x147f); // stopped by SIGTSTP (18 -> 20)
        assert_eq!(wait_status_d2l(0x137f), 0xffff); // Darwin "continued" -> Linux encoding
    }

    #[test]
    fn open_flags_translate_or_refuse() {
        assert_eq!(open_flags_l2d(0o1101), Some(0x1 | 0x200 | 0x400)); // O_WRONLY|O_CREAT|O_TRUNC
        assert_eq!(open_flags_l2d(0o2000000), Some(0x100_0000));
        assert_eq!(open_flags_l2d(0o40000), Some(0x10_0000));
        assert_eq!(open_flags_l2d(0o20000000), None); // O_PATH has no Darwin equivalent
    }

    #[test]
    fn ctype_table_matches_glibc_c_locale() {
        assert_eq!(
            glibc_class(b'A'),
            0x100 | 0x400 | 0x1000 | 0x4000 | 0x8000 | 0x8
        );
        assert_eq!(glibc_class(b' '), 0x2000 | 0x4000 | 0x1);
        assert_eq!(glibc_class(b'\n'), 0x2000 | 0x2);
        assert_eq!(glibc_class(b'5'), 0x800 | 0x1000 | 0x4000 | 0x8000 | 0x8);
        assert_eq!(glibc_class(b'!'), 0x4 | 0x4000 | 0x8000);
        assert_eq!(signal_l2d(7), 10);
        assert_eq!(signal_l2d(16), 0);
    }
}
