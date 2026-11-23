// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: isolated worker process for NVIDIA's Linux-aarch64
// `tileiras`. NVIDIA code is executed ONLY by this helper, never by the
// planner, by `cargo test` itself or by a long-lived tool. The helper
//   * admits exactly one pinned ELF and one pinned patch manifest;
//   * maps the image without overwriting any existing mapping (image.rs);
//   * arms a wall-clock alarm and a CPU limit before the first guest
//     instruction; the guest has no `svc` instruction (manifest: svc=0) and no
//     socket import, so every kernel interaction goes through shim.rs;
//   * `--map-check` and `--unwind-gate-check` stop after mapping and execute
//     no NVIDIA instruction;
//   * the guest's working directory AND its TMPDIR are the scratch directory;
//     the only child process it can start is the pinned native ptxas.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("tileiras-native-helper: macOS arm64 only");
    std::process::exit(2);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod dso;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod image;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod shim;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod worker {
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    use tileiras_native::{patches, sha256};

    use crate::{image, shim};

    pub const ELF_SHA256: &str = "297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a";
    /// scripts/gen_patch_manifest.py is deterministic, so its output is pinned too.
    pub const MANIFEST_SHA256: &str =
        "9752c4f79fcc46640af1d2f293c2135c6c9d7d3cad2caa7c014f85b43782fb84";
    const DEFAULT_ELF: &str = ".config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/cuda_tileiras-linux-sbsa-13.2.51-archive/bin/tileiras";
    const DEFAULT_MANIFEST: &str = ".cache/tileiras-native/13.2.51/patches.tsv";
    const DEFAULT_LIBNVVM: &str = ".config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/libnvvm-linux-sbsa-13.2.51-archive/nvvm/lib64/libnvvm.so.4.0.0";
    const DEFAULT_LIBNVVM_MANIFEST: &str = ".cache/tileiras-native/13.2.51/libnvvm.patches.tsv";

    extern "C" {
        fn pthread_key_create(k: *mut u64, d: *mut c_void) -> c_int;
        fn pthread_setspecific(k: u64, v: *mut c_void) -> c_int;
        fn alarm(seconds: u32) -> u32;
        fn setrlimit(resource: c_int, lim: *const u64) -> c_int;
        fn sigaction(sig: c_int, new: *const SigAction, old: *mut SigAction) -> c_int;
        fn sigaltstack(new: *const [usize; 3], old: *mut [usize; 3]) -> c_int;
        fn write(fd: c_int, buf: *const u8, n: usize) -> isize;
        fn _exit(code: c_int) -> !;
        fn chdir(path: *const c_char) -> c_int;
    }

    #[repr(C)]
    struct SigAction {
        handler: usize,
        mask: u32,
        flags: c_int,
    }

    static mut IMAGE_LO: u64 = 0;
    static mut IMAGE_HI: u64 = 0;
    static mut RESERVE_END: u64 = 0;

    /// Fatal-signal reporter. Nothing is mapped below 4 GiB, so a link-time
    /// pointer that escaped rebasing shows up here as a fault at its own value.
    unsafe extern "C" fn on_fault(sig: c_int, info: *const u8, context: *const u8) {
        let slide = image::SLIDE;
        let addr = *(info.add(24) as *const u64);
        let mc = *(context.add(48) as *const *const u64);
        let (pc, lr) = (*mc.add(2 + 32), *mc.add(2 + 30));
        let (lo, hi, end) = (IMAGE_LO, IMAGE_HI, RESERVE_END);
        let place = |a: u64| -> String {
            let (dso_lo, dso_hi) = (
                shim::DSO_BASE.load(Ordering::Relaxed),
                shim::DSO_END.load(Ordering::Relaxed),
            );
            if a >= dso_lo && a < dso_hi {
                format!("libnvvm link-time {:#x}", a - dso_lo)
            } else if a >= lo + slide && a < hi + slide {
                format!("guest link-time {:#x}", a - slide)
            } else if a >= hi + slide && a < end {
                "helper island/thunk".to_string()
            } else {
                "outside the guest image".to_string()
            }
        };
        let mut text = format!(
            "tileiras-native-helper: fatal signal {sig} at pc={pc:#x} ({}) lr={lr:#x} ({}) fault address {addr:#x}\n",
            place(pc), place(lr)
        );
        if (lo..hi).contains(&addr) || (lo..hi).contains(&pc) {
            text.push_str("tileiras-native-helper: the faulting value is an UNREBASED link-time address (a pointer the manifest's rebase rule missed)\n");
        } else if let Some(name) = shim::loaded_dso().and_then(|d| {
            // libNVVM's data-import traps first: their addresses lie inside
            // the range tested below.
            d.data_traps
                .range(..=addr)
                .next_back()
                .filter(|(a, _)| addr - **a < 16)
        }) {
            text.push_str(&format!(
                "tileiras-native-helper: access to unimplemented libNVVM data import '{}'\n",
                name.1
            ));
        } else if (0xdead_0000u64..0xdead_0000 + (1 << 20)).contains(&addr) {
            // try_lock: the fault may have interrupted a holder of the lock.
            let names = shim::IMPORT_NAMES
                .try_lock()
                .map(|n| n.clone())
                .unwrap_or_default();
            let name = names
                .get(((addr - 0xdead_0000) / 16) as usize)
                .cloned()
                .unwrap_or_default();
            text.push_str(&format!(
                "tileiras-native-helper: access to unimplemented data import '{name}'\n"
            ));
        }
        write(2, text.as_ptr(), text.len());
        _exit(71)
    }

    /// Main thread only (guest worker threads have no alternate stack); the
    /// report allocates, so a fault inside malloc may lose the message but the
    /// process still dies from the repeated fault.
    unsafe fn install_fault_reporter() -> Result<(), String> {
        let size = 256 * 1024;
        let stack = Box::leak(vec![0u8; size].into_boxed_slice());
        if sigaltstack(
            &[stack.as_mut_ptr() as usize, size, 0],
            std::ptr::null_mut(),
        ) != 0
        {
            return Err("sigaltstack failed".into());
        }
        for sig in [4, 5, 8, 10, 11] {
            let action = SigAction {
                handler: on_fault as *const () as usize,
                mask: 0,
                flags: 0x40 | 0x1, // SA_SIGINFO | SA_ONSTACK
            };
            if sigaction(sig, &action, std::ptr::null_mut()) != 0 {
                return Err("sigaction failed".into());
            }
        }
        Ok(())
    }

    fn json_string(value: &str) -> String {
        let mut out = String::from("\"");
        for c in value.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }
    fn json_list(values: &[String]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(|v| json_string(v))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn help() {
        println!(
            r#"tileiras-native-helper: isolated macOS arm64 worker for NVIDIA's Linux-aarch64 tileiras

Usage:
  tileiras-native-helper --map-check [--elf FILE] [--manifest FILE]
  tileiras-native-helper --nvvm-check [--libnvvm FILE] [--libnvvm-manifest FILE]
  tileiras-native-helper --unwind-gate-check island|native
  tileiras-native-helper [--elf FILE] [--manifest FILE] [--timeout SECONDS]
                         [--scratch DIR] -- <tileiras arguments>

--nvvm-check  Load the pinned linux-sbsa libnvvm.so.4 as the guest's dlopen would,
              run its constructors and call nvvmVersion (executes NVIDIA code).
--ptxas-resolver FILE  tools/ptxas-resolve.sh: when the guest spawns `ptxas`, the
              verified native ptxas it prints is run instead. Without it, and
              for every other program, a spawn is a named trap (exit 70).
--unwind-gate-check island|native
              Map the image, then ask the dl_iterate_phdr shim (as the guest's
              C++ unwinder would) for a pc in guest code, in the guest stack
              root and in non-code memory (all three return), and finally in an
              import thunk or a native function: that one must end the process
              with the named diagnostic and exit 72. Executes NO NVIDIA code.
--env K=V     Add one variable to the guest's explicit environment (PATH and
              TMPDIR are set by the helper and cannot be replaced).
--map-check   Map, relocate, patch and protect the image, verify the page
              protections with mach_vm_region, print a JSON receipt, exit.
              Executes NO NVIDIA instruction.
-- ARGS       Run the guest with ARGS. Unimplemented imports exit 70 naming the
              symbol; guest faults exit 71; a C++ exception that would have to
              cross a helper frame exits 72; the alarm (default 60 s) kills it.
              Working directory and TMPDIR = --scratch DIR (default: cwd).
--helper-version  Version of this helper (the guest's --version is `-- --version`)."#
        );
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args().skip(1);
        let (mut elf, mut manifest, mut map_check) = (None, None, false);
        let (mut timeout, mut scratch, mut guest) = (60u32, None, None::<Vec<String>>);
        let (mut libnvvm, mut libnvvm_manifest, mut nvvm_check) = (None, None, false);
        let mut extra_env: Vec<String> = Vec::new();
        let mut gate_check = None::<String>;
        while let Some(a) = args.next() {
            match a.as_str() {
                "-h" | "--help" => {
                    help();
                    return Ok(());
                }
                "--helper-version" => {
                    println!("tileiras-native-helper 0.1.0 (loader; not NVIDIA's version string)");
                    return Ok(());
                }
                "--map-check" => map_check = true,
                "--nvvm-check" => nvvm_check = true,
                "--unwind-gate-check" => {
                    gate_check = Some(
                        args.next()
                            .filter(|v| v == "island" || v == "native")
                            .ok_or("--unwind-gate-check needs island or native")?,
                    )
                }
                "--libnvvm" => {
                    libnvvm = Some(PathBuf::from(args.next().ok_or("--libnvvm needs a file")?))
                }
                "--libnvvm-manifest" => {
                    libnvvm_manifest = Some(PathBuf::from(
                        args.next().ok_or("--libnvvm-manifest needs a file")?,
                    ))
                }
                "--ptxas-resolver" => {
                    // Made absolute NOW: the helper changes directory to the
                    // scratch directory later, and a relative path must not
                    // start naming a file there.
                    let given = PathBuf::from(args.next().ok_or("--ptxas-resolver needs a file")?);
                    let absolute = std::fs::canonicalize(&given)
                        .map_err(|e| format!("--ptxas-resolver {}: {e}", given.display()))?;
                    if !absolute.is_file() {
                        return Err(format!(
                            "--ptxas-resolver {} is not a file",
                            absolute.display()
                        ));
                    }
                    *shim::PTXAS_RESOLVER.lock().unwrap() = Some(absolute)
                }
                "--env" => {
                    let kv = args
                        .next()
                        .filter(|v| {
                            v.contains('=')
                                && !v.starts_with('=')
                                && !v.starts_with("PATH=")
                                && !v.starts_with("TMPDIR=")
                        })
                        .ok_or("--env needs KEY=VALUE (not PATH or TMPDIR)")?;
                    extra_env.push(kv);
                }
                "--elf" => elf = Some(PathBuf::from(args.next().ok_or("--elf needs a file")?)),
                "--manifest" => {
                    manifest = Some(PathBuf::from(args.next().ok_or("--manifest needs a file")?))
                }
                "--timeout" => {
                    timeout = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .filter(|v| (1..=3600).contains(v))
                        .ok_or("--timeout needs 1..3600 seconds")?
                }
                "--scratch" => scratch = Some(args.next().ok_or("--scratch needs a directory")?),
                "--" => {
                    guest = Some(args.by_ref().collect());
                    break;
                }
                other => {
                    return Err(format!(
                        "unknown option {other} (guest arguments go after --)"
                    ))
                }
            }
        }
        if [map_check, nvvm_check, gate_check.is_some(), guest.is_some()]
            .iter()
            .filter(|x| **x)
            .count()
            != 1
        {
            return Err("choose exactly one of --map-check, --nvvm-check, --unwind-gate-check or -- <tileiras arguments>".into());
        }
        let home = PathBuf::from(std::env::var_os("HOME").ok_or("HOME missing")?);
        let elf_path = elf.unwrap_or_else(|| home.join(DEFAULT_ELF));
        let manifest_path = manifest
            .or_else(|| std::env::var_os("TILEIRAS_NATIVE_MANIFEST").map(PathBuf::from))
            .unwrap_or_else(|| home.join(DEFAULT_MANIFEST));
        let bytes = std::fs::read(&elf_path).map_err(|e| format!("{}: {e}", elf_path.display()))?;
        if bytes.len() > tileiras_native::MAX_FILE_BYTES || sha256::hex(&bytes) != ELF_SHA256 {
            return Err(format!(
                "{} is not the admitted tileiras 13.2.51 binary",
                elf_path.display()
            ));
        }
        let text = std::fs::read(&manifest_path).map_err(|e| {
            format!(
                "{}: {e} (run portkit/tileiras-native/build.sh)",
                manifest_path.display()
            )
        })?;
        if sha256::hex(&text) != MANIFEST_SHA256 {
            return Err(format!(
                "{} is not the pinned patch manifest",
                manifest_path.display()
            ));
        }
        let manifest = patches::parse(std::str::from_utf8(&text).map_err(|e| e.to_string())?)?;
        let plan = tileiras_native::plan(&bytes, image::PAGE).map_err(|e| e.to_string())?;
        let tls_bytes = plan.tls.as_ref().map_or(0, |t| t.memory_size);
        if plan
            .tls
            .as_ref()
            .is_some_and(|t| !t.initialized_bytes.is_empty() || t.alignment > 16)
        {
            return Err("only a zero-initialised PT_TLS with alignment <= 16 is supported".into());
        }

        // Two TSD slots, read by the islands relative to TPIDRRO_EL0. Check the
        // "pthread key N lives at tsd[N]" layout the islands rely on.
        let (mut tp_key, mut x18_key) = (0u64, 0u64);
        unsafe {
            // The destructors put the value back: Darwin clears a slot before
            // running later keys' destructors, and guest key destructors
            // (created later, so run later) may still use TLS and x18.
            if pthread_key_create(&mut tp_key, shim::keep_tp as *mut c_void) != 0
                || pthread_key_create(&mut x18_key, shim::keep_x18 as *mut c_void) != 0
            {
                return Err("pthread_key_create failed".into());
            }
            for key in [tp_key, x18_key] {
                let sentinel = 0x7469_6c65_0000_0000u64 | key;
                pthread_setspecific(key, sentinel as *mut c_void);
                let base: u64;
                std::arch::asm!("mrs {t}, tpidrro_el0", t = out(reg) base);
                let seen = *(((base & !7) + 8 * key) as *const u64);
                pthread_setspecific(key, std::ptr::null_mut());
                if seen != sentinel || key > 4095 {
                    return Err("this libpthread does not keep TSD at TPIDRRO_EL0[key]".into());
                }
            }
        }
        shim::TP_KEY.store(tp_key, Ordering::SeqCst);
        shim::X18_KEY.store(x18_key, Ordering::SeqCst);
        // Static TLS: the executable's block, then room for libNVVM's.
        let exe_tls = (tls_bytes + 15) & !15;
        shim::TLS_BYTES.store(exe_tls + 256, Ordering::SeqCst);
        *shim::DSO_CONFIG.lock().unwrap() = Some(shim::DsoConfig {
            path: libnvvm.unwrap_or_else(|| home.join(DEFAULT_LIBNVVM)),
            manifest: libnvvm_manifest.unwrap_or_else(|| home.join(DEFAULT_LIBNVVM_MANIFEST)),
            tp_slot: 8 * tp_key,
            x18_slot: 8 * x18_key,
            tls_offset: 16 + exe_tls,
            tls_capacity: 256,
        });
        shim::SLIDE.store(image::SLIDE, Ordering::SeqCst);
        shim::PHDR.store(
            plan.startup.program_headers_address + image::SLIDE,
            Ordering::SeqCst,
        );
        shim::PHNUM.store(plan.startup.program_header_count as u64, Ordering::SeqCst);
        shim::ENTRY.store(plan.entry + image::SLIDE, Ordering::SeqCst);
        *shim::IMPORT_NAMES.lock().unwrap() = plan.imports.iter().map(|i| i.name.clone()).collect();

        let loaded = image::load(&image::Inputs {
            elf: &bytes,
            plan: &plan,
            manifest: &manifest,
            tp_slot: 8 * tp_key,
            x18_slot: 8 * x18_key,
            resolve: &shim::resolve,
            trap_handler: shim::trap_unimplemented as *const () as usize as u64,
            errno_sync: shim::errno_sync as *const () as usize as u64,
        })?;
        let regions = image::verify(&loaded)?;
        let island_start = loaded
            .regions
            .last()
            .map_or(loaded.reserve_end, |r| r.start);
        shim::GUEST_LO.store(loaded.reserve_start, Ordering::SeqCst);
        shim::GUEST_HI.store(island_start, Ordering::SeqCst);
        shim::ISLAND_END.store(loaded.reserve_end, Ordering::SeqCst);
        unsafe {
            IMAGE_LO = manifest.image_lo;
            IMAGE_HI = manifest.image_hi;
            RESERVE_END = loaded.reserve_end;
        }

        if map_check {
            let prot = |p: i32| {
                format!(
                    "{}{}{}",
                    if p & 1 != 0 { "r" } else { "-" },
                    if p & 2 != 0 { "w" } else { "-" },
                    if p & 4 != 0 { "x" } else { "-" }
                )
            };
            let regions_json = regions
                .iter()
                .map(|(name, start, end, want, got)| {
                    format!("{{\"name\":{},\"start\":{start},\"end\":{end},\"wanted\":\"{}\",\"mach_vm_region\":\"{}\"}}", json_string(name), prot(*want), prot(*got))
                })
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "{{\"schema\":\"cuda-oxide.tileiras-native-map-check.v1\",\"elf_sha256\":\"{}\",\"manifest_sha256\":\"{}\",\"link_base\":{},\"fixed_link_base_mappable\":false,\"slide\":{},\"reserved_start\":{},\"reserved_end\":{},\"reservation\":\"mach_vm_allocate VM_FLAGS_FIXED without VM_FLAGS_OVERWRITE\",\"regions\":[{}],\"writable_and_executable_pages\":0,\"relro_writable_tail_link_addresses\":[{},{}],\"rebased_data_pointers\":{},\"contradicted_words_left_unslid\":{},\"x18_sites_patched\":{},\"tls_sites_patched\":{},\"imports_total\":{},\"imports_bound\":{},\"imports_weak_null\":{},\"imports_trapping\":{},\"tls_bytes_per_thread\":{},\"nvidia_code_executed\":false}}",
                ELF_SHA256, MANIFEST_SHA256, plan.load_base, loaded.slide, loaded.reserve_start, loaded.reserve_end,
                regions_json, loaded.relro_writable_tail.0, loaded.relro_writable_tail.1, loaded.rebased, loaded.not_rebased, loaded.x18_sites, loaded.tls_sites, loaded.import_names.len(),
                loaded.imports_bound.len(), json_list(&loaded.imports_weak_null), json_list(&loaded.imports_trapping), tls_bytes
            );
            return Ok(());
        }

        if let Some(kind) = gate_check {
            let on_stack = 0u64;
            unsafe {
                // These three are the unwinder's own business: the shim returns.
                shim::unwind_gate_probe(loaded.entry);
                shim::unwind_gate_probe(shim::guest_root_pc() - 1);
                shim::unwind_gate_probe(&on_stack as *const u64 as u64);
                println!("{{\"schema\":\"cuda-oxide.tileiras-native-unwind-gate-check.v1\",\"guest_pc_returned\":true,\"stack_root_pc_returned\":true,\"non_code_pc_returned\":true,\"probing\":{},\"nvidia_code_executed\":false}}", json_string(&kind));
                use std::io::Write;
                std::io::stdout().flush().ok();
                // This one must not return.
                shim::unwind_gate_probe(if kind == "island" {
                    island_start + 16
                } else {
                    write as *const () as u64 + 4
                });
            }
            return Err("the unwind gate let a helper frame through".into());
        }

        if nvvm_check {
            // Load libNVVM, run its constructors and ask it for its version.
            // This executes NVIDIA code, hence the same alarm as a guest run.
            unsafe {
                install_fault_reporter()?;
                shim::install_thread_block();
                alarm(timeout);
                shim::clear_host_errno();
                let handle = shim::sh_dlopen(c"libnvvm.so".as_ptr(), 0);
                if handle.is_null() {
                    return Err("libNVVM could not be loaded".into());
                }
                let f = shim::sh_dlsym(handle, c"nvvmVersion".as_ptr());
                if f.is_null() {
                    return Err("nvvmVersion is not exported".into());
                }
                let f: extern "C" fn(*mut c_int, *mut c_int) -> c_int = std::mem::transmute(f);
                let (mut major, mut minor) = (-1, -1);
                let rc = f(&mut major, &mut minor);
                let d = shim::loaded_dso().unwrap();
                println!(
                    "{{\"schema\":\"cuda-oxide.tileiras-native-nvvm-check.v1\",\"libnvvm_sha256\":\"{}\",\"manifest_sha256\":\"{}\",\"base\":{},\"constructors_run\":{},\"relocations\":{{{}}},\"x18_sites_patched\":{},\"tls_sites_patched\":{},\"tls_bytes\":{},\"exports\":{},\"imports_bound\":{},\"imports_trapping\":{},\"nvvmVersion_result\":{rc},\"nvvm_major\":{major},\"nvvm_minor\":{minor},\"nvidia_code_executed\":true}}",
                    shim::LIBNVVM_SHA256, shim::LIBNVVM_MANIFEST_SHA256, d.base, d.init.len(),
                    d.relocations.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(","),
                    d.x18_sites, d.tls_sites, d.tls_bytes, d.exports.len(), d.imports_bound, json_list(&d.imports_trapping)
                );
                return if rc == 0 {
                    Ok(())
                } else {
                    Err("nvvmVersion failed".into())
                };
            }
        }

        // ---- run the guest ---------------------------------------------------
        let guest = guest.unwrap();
        if let Some(dir) = scratch {
            let c = CString::new(dir).map_err(|e| e.to_string())?;
            if unsafe { chdir(c.as_ptr()) } != 0 {
                return Err("cannot enter the scratch directory".into());
            }
        }
        // Temporary files (PTX, ptxas output, logs) stay in the working
        // directory, never in the host's TMPDIR.
        let work = std::env::current_dir().map_err(|e| e.to_string())?;
        let work = work
            .to_str()
            .ok_or("the working directory is not UTF-8")?
            .to_string();
        // tileiras looks `ptxas` up in PATH before it spawns it. With a
        // resolver, PATH starts with a directory holding a placeholder of that
        // name; it is never executed (the spawn shim runs the pinned native
        // ptxas instead) and fails loudly if anything else ever runs it.
        let mut path = std::env::var("PATH").unwrap_or_default();
        if shim::PTXAS_RESOLVER.lock().unwrap().is_some() {
            let dir = PathBuf::from(&work).join(".tileiras-native-bin");
            place_ptxas_placeholder(&dir)?;
            path = format!("{}:{path}", dir.display());
        }
        let mut strings: Vec<CString> = vec![CString::new("tileiras").unwrap()];
        for a in &guest {
            strings.push(CString::new(a.as_str()).map_err(|e| e.to_string())?);
        }
        let argc = strings.len();
        // A small, explicit environment: nothing of the host's leaks in by default.
        strings.push(CString::new(format!("PATH={path}")).map_err(|e| e.to_string())?);
        strings.push(CString::new(format!("TMPDIR={work}")).map_err(|e| e.to_string())?);
        for key in ["HOME", "TERM"] {
            if let Ok(v) = std::env::var(key) {
                strings.push(CString::new(format!("{key}={v}")).map_err(|e| e.to_string())?);
            }
        }
        for kv in &extra_env {
            strings.push(CString::new(kv.as_str()).map_err(|e| e.to_string())?);
        }
        strings.push(CString::new("LANG=C").unwrap());
        strings.push(CString::new("LC_ALL=C").unwrap());
        let mut block: Vec<u64> = vec![argc as u64];
        block.extend(strings[..argc].iter().map(|s| s.as_ptr() as u64));
        block.push(0);
        block.extend(strings[argc..].iter().map(|s| s.as_ptr() as u64));
        block.push(0);
        for (k, v) in shim::auxv() {
            block.extend([k, v]);
        }
        block.extend([0, 0]);
        if block.len() % 2 != 0 {
            block.push(0);
        }
        std::mem::forget(strings);
        unsafe {
            install_fault_reporter()?;
            shim::clear_host_errno();
            shim::install_thread_block();
            let cpu = [timeout as u64, timeout as u64 + 5];
            setrlimit(0, cpu.as_ptr()); // RLIMIT_CPU
            alarm(timeout);
            enter(loaded.entry, block.as_ptr(), block.len())
        }
    }

    const PLACEHOLDER: &str = "#!/bin/sh\necho 'tileiras-native-helper placeholder: ptxas spawns are intercepted; this file must never run' >&2\nexit 97\n";

    /// Create `<dir>/ptxas` without ever writing through a symbolic link: the
    /// directory must be a real directory, the file is created with O_EXCL,
    /// and an existing entry is accepted only if it is a regular file with
    /// exactly the placeholder's content (a concurrent helper's).
    fn place_ptxas_placeholder(dir: &std::path::Path) -> Result<(), String> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::create_dir(dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        }
        if !std::fs::symlink_metadata(dir).is_ok_and(|m| m.file_type().is_dir()) {
            return Err(format!("{} exists and is not a directory", dir.display()));
        }
        let stub = dir.join("ptxas");
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o755)
            .open(&stub)
        {
            Ok(mut f) => f
                .write_all(PLACEHOLDER.as_bytes())
                .map_err(|e| format!("{}: {e}", stub.display())),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let regular =
                    std::fs::symlink_metadata(&stub).is_ok_and(|m| m.file_type().is_file());
                if regular && std::fs::read(&stub).is_ok_and(|b| b == PLACEHOLDER.as_bytes()) {
                    Ok(())
                } else {
                    Err(format!(
                        "{} exists and is not the helper's placeholder; remove it",
                        stub.display()
                    ))
                }
            }
            Err(e) => Err(format!("{}: {e}", stub.display())),
        }
    }

    /// Copy the Linux initial stack image (argc, argv, envp, auxv) to the top of
    /// the current stack and jump to the ELF entry point, as the kernel would.
    unsafe fn enter(entry: u64, block: *const u64, words: usize) -> ! {
        std::arch::asm!(
            "lsl x3, x2, #3",
            "sub x4, sp, x3",
            "and x4, x4, #0xfffffffffffffff0",
            "mov sp, x4",
            "2:",
            "cbz x2, 3f",
            "ldr x5, [x1], #8",
            "str x5, [x4], #8",
            "sub x2, x2, #1",
            "b 2b",
            "3:",
            "mov x0, #0",
            "mov x29, #0",
            "mov x30, #0",
            "br x9",
            in("x9") entry,
            in("x1") block,
            in("x2") words,
            options(noreturn)
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = worker::run() {
        eprintln!("tileiras-native-helper: {error}");
        std::process::exit(2);
    }
}
