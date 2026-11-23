// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: gates for tileiras-native-helper. NVIDIA code runs
// only inside the spawned helper process, never in this test process; every
// spawn has a hard wall-clock limit and a scratch working directory under the
// cargo target directory. Tests that need the pinned NVIDIA ELF are ignored by
// default (set TILEIRAS_NATIVE_TEST_ELF and pass --include-ignored).
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const HELPER: &str = env!("CARGO_BIN_EXE_tileiras-native-helper");

struct Outcome {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run the helper with a hard 60 s wall clock; a hang is a test failure.
fn helper(args: &[&str], envs: &[(&str, &str)], cwd: &PathBuf) -> Outcome {
    let mut command = Command::new(HELPER);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in envs {
        command.env(k, v);
    }
    bounded(command, &format!("{args:?}"))
}

/// Run a child with a hard 60 s wall clock; a hang is a test failure.
fn bounded(mut command: Command, what: &str) -> Outcome {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    // Drain both pipes while waiting so a chatty child cannot block on a full pipe.
    let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let out_reader = std::thread::spawn(move || {
        let mut s = String::new();
        out.read_to_string(&mut s).ok();
        s
    });
    let err_reader = std::thread::spawn(move || {
        let mut s = String::new();
        err.read_to_string(&mut s).ok();
        s
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("child exceeded the 60 s wall clock: {what}");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Outcome {
        code: status.code(),
        stdout: out_reader.join().unwrap(),
        stderr: err_reader.join().unwrap(),
    }
}

fn pinned_elf() -> String {
    std::env::var("TILEIRAS_NATIVE_TEST_ELF")
        .expect("set TILEIRAS_NATIVE_TEST_ELF to the SHA256-pinned NVIDIA tileiras")
}
fn guest(args: &[&str], dir: &PathBuf) -> Outcome {
    let elf = pinned_elf();
    let mut all = vec!["--elf", &elf, "--timeout", "45", "--"];
    all.extend_from_slice(args);
    helper(&all, &[], dir)
}

#[test]
fn helper_cli_refuses_everything_but_one_explicit_mode() {
    let dir = scratch("cli");
    let help = helper(&["--help"], &[], &dir);
    assert_eq!(help.code, Some(0));
    assert!(help.stdout.contains("Executes NO NVIDIA instruction"));
    for args in [
        &[][..],
        &["--map-check", "--", "--version"][..],
        &["--version"][..],
        &["--run"][..],
        &["--timeout", "0", "--", "--version"][..],
    ] {
        let out = helper(args, &[], &dir);
        assert_eq!(out.code, Some(2), "{args:?}: {}", out.stderr);
        assert!(out.stdout.is_empty());
    }
}

#[test]
fn helper_refuses_an_unpinned_elf_before_mapping_anything() {
    let dir = scratch("unpinned");
    let fake = dir.join("tileiras");
    std::fs::write(&fake, b"\x7fELF not the admitted binary").unwrap();
    for mode in [&["--map-check"][..], &["--", "--version"][..]] {
        let mut args = vec!["--elf", fake.to_str().unwrap()];
        args.extend_from_slice(mode);
        let out = helper(&args, &[], &dir);
        assert_eq!(out.code, Some(2));
        assert!(
            out.stderr.contains("not the admitted tileiras"),
            "{}",
            out.stderr
        );
    }
}

fn number(json: &str, key: &str) -> u64 {
    let at = json
        .find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("{key} missing"))
        + key.len()
        + 3;
    json[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap()
}

#[test]
#[ignore = "requires the pinned ELF and patch manifest: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn map_check_reports_kernel_verified_protections_and_executes_nothing() {
    let elf = pinned_elf();
    let out = helper(&["--map-check", "--elf", &elf], &[], &scratch("map"));
    assert_eq!(out.code, Some(0), "{}", out.stderr);
    let j = out.stdout.trim();
    assert!(j.contains("\"nvidia_code_executed\":false"));
    assert!(j.contains("\"fixed_link_base_mappable\":false"));
    assert!(j.contains("\"writable_and_executable_pages\":0"));
    let slide = 0x70_0000_0000u64;
    for (name, start, end, prot) in [
        ("text", 0x400000u64, 0x4d64000u64, "r-x"),
        ("guard", 0x4d64000, 0x4d74000, "---"),
        ("relro", 0x4d74000, 0x5940000, "r--"),
        ("data", 0x5940000, 0x5b1c000, "rw-"),
    ] {
        let region = format!(
            "\"name\":\"{name}\",\"start\":{},\"end\":{},\"wanted\":\"{prot}\",\"mach_vm_region\":\"{prot}\"",
            start + slide,
            end + slide
        );
        assert!(j.contains(&region), "missing {region} in {j}");
    }
    assert!(j.contains(&format!(
        "\"name\":\"islands\",\"start\":{}",
        0x5b1c000 + slide
    )));
    assert_eq!(number(j, "slide"), 0x70_0000_0000);
    assert_eq!(number(j, "x18_sites_patched"), 9543);
    assert_eq!(number(j, "tls_sites_patched"), 592);
    assert_eq!(number(j, "imports_total"), 276);
    assert_eq!(number(j, "rebased_data_pointers"), 134_750);
    assert_eq!(number(j, "contradicted_words_left_unslid"), 157);
    assert_eq!(number(j, "tls_bytes_per_thread"), 772);
    // Process creation other than the intercepted posix_spawn stays a named trap.
    let list = &j[j.find("\"imports_trapping\":").unwrap()..];
    for name in ["fork", "execve", "execv", "kill", "setsid", "poll"] {
        assert!(list.contains(&format!("\"{name}\"")), "{name} must trap");
    }
    for name in ["posix_spawn", "wait4", "dlopen", "socket", "connect"] {
        assert!(
            !list.contains(&format!("\"{name}\"")),
            "{name} is bound or not imported at all"
        );
    }
}

fn repo_file(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// The data-pointer rebase rule, cross-checked two ways: the Python generator
/// that wrote the manifest and the independent Rust audit must agree on the
/// exact set of words left unslid, and the class counts are pinned.
#[test]
#[ignore = "requires the pinned ELF and patch manifest: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn rebase_rule_is_cross_checked_and_its_uncorroborated_part_is_counted() {
    use tileiras_native::patches::{self, PatchKind};
    let elf = std::fs::read(pinned_elf()).unwrap();
    let audit = tileiras_native::rebase_audit::audit(&elf).unwrap();
    let home = std::env::var("HOME").unwrap();
    let manifest_path = std::env::var("TILEIRAS_NATIVE_MANIFEST")
        .unwrap_or_else(|_| format!("{home}/.cache/tileiras-native/13.2.51/patches.tsv"));
    let manifest = patches::parse(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
    let unslid: Vec<(u64, u64)> = manifest
        .rows
        .iter()
        .filter(|r| r.kind == PatchKind::NoRebase)
        .map(|r| (r.address, (r.new_word as u64) << 32 | r.old_word as u64))
        .collect();
    assert_eq!(
        unslid, audit.contradicted,
        "generator and audit disagree on the contradicted words"
    );
    assert_eq!(
        manifest.rebase.iter().map(|r| r.count).sum::<u64>(),
        audit.rebased_total
    );
    for range in &manifest.rebase {
        assert_eq!(
            audit.by_container[&range.section], range.count,
            "{}",
            range.section
        );
    }

    assert_eq!(audit.candidates_total, 134_907);
    assert_eq!(audit.contradicted.len(), 157);
    let reasons: Vec<u64> = tileiras_native::rebase_audit::CONTRADICTIONS
        .iter()
        .map(|k| audit.contradicted_by_reason[k])
        .collect();
    assert_eq!(reasons, [4, 8, 28, 117]); // outside sections, unwind metadata, unaligned, mid-function
    assert_eq!(audit.contradicted_but_code_formed, 0);
    assert_eq!(audit.rebased_total, 134_750);
    let classes: Vec<u64> = tileiras_native::rebase_audit::CLASSES
        .iter()
        .map(|k| audit.by_class[k])
        .collect();
    // pointer-only section, function start, code address target,
    // pointer-bearing object, C string start, ambiguous
    assert_eq!(classes, [22_650, 97_130, 1_597, 1_909, 7_588, 3_876]);
    assert_eq!(classes.iter().sum::<u64>(), audit.rebased_total);
    assert_eq!(audit.ambiguous, 3_876);
    // No slid value in .data.rel.ro / .data points into code without being a
    // function start; the two that do are GOT entries.
    assert_eq!(audit.executable_target_not_function_start, 2);
    assert_eq!(audit.not_rebased_in_range[".rodata"], 14_658);
    let json = audit.to_json("x");
    assert!(json.contains("\"ambiguous\":3876") && json.contains("\"nvidia_code_executed\":false"));
}

#[test]
#[ignore = "requires the pinned ELF and patch manifest: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn an_exception_that_would_cross_a_helper_frame_ends_with_a_named_diagnostic() {
    let elf = pinned_elf();
    for (kind, needle) in [
        (
            "island",
            "is an import thunk / instruction island of the helper",
        ),
        (
            "native",
            "is native function write in /usr/lib/system/libsystem_kernel.dylib",
        ),
    ] {
        let out = helper(
            &["--elf", &elf, "--unwind-gate-check", kind],
            &[],
            &scratch("gate"),
        );
        assert_eq!(out.code, Some(72), "{kind}: {}", out.stderr);
        // Guest pcs, the guest stack root and non-code words were passed through first.
        assert!(out.stdout.contains("\"guest_pc_returned\":true,\"stack_root_pc_returned\":true,\"non_code_pc_returned\":true"), "{}", out.stdout);
        assert!(out.stdout.contains("\"nvidia_code_executed\":false"));
        assert!(
            out.stderr
                .contains("C++ exception cannot unwind through a non-guest frame"),
            "{}",
            out.stderr
        );
        assert!(out.stderr.contains(needle), "{}", out.stderr);
    }
}

#[test]
#[ignore = "executes NVIDIA code in the helper process: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn libnvvm_is_loaded_relocated_initialised_and_answers_nvvm_version() {
    let elf = pinned_elf();
    let out = helper(
        &["--elf", &elf, "--timeout", "45", "--nvvm-check"],
        &[],
        &scratch("nvvm"),
    );
    assert_eq!(out.code, Some(0), "{}", out.stderr);
    let j = out.stdout.trim();
    assert!(j.contains("\"relocations\":{\"ABS64\":1,\"GLOB_DAT\":37,\"JUMP_SLOT\":242,\"RELATIVE\":65712,\"TLSDESC\":4}"), "{j}");
    assert_eq!(number(j, "constructors_run"), 741);
    assert_eq!(number(j, "x18_sites_patched"), 7873);
    assert_eq!(number(j, "tls_sites_patched"), 390);
    assert_eq!(number(j, "tls_bytes"), 40);
    assert_eq!(number(j, "nvvmVersion_result"), 0);
    assert_eq!((number(j, "nvvm_major"), number(j, "nvvm_minor")), (2, 0));
    // libNVVM's process-creation and network-adjacent imports stay named traps.
    let list = &j[j.find("\"imports_trapping\":").unwrap()..];
    for name in ["fork", "execve", "execv", "pipe", "kill", "select", "poll"] {
        assert!(list.contains(&format!("\"{name}\"")), "{name} must trap");
    }
    // A tampered libNVVM manifest is refused before the object is mapped.
    let dir = scratch("nvvm");
    let home = std::env::var("HOME").unwrap();
    let mut bytes = std::fs::read(format!(
        "{home}/.cache/tileiras-native/13.2.51/libnvvm.patches.tsv"
    ))
    .unwrap();
    let last = bytes.len() - 2;
    bytes[last] ^= 1;
    let tampered = dir.join("libnvvm.patches.tsv");
    std::fs::write(&tampered, bytes).unwrap();
    let out = helper(
        &[
            "--elf",
            &elf,
            "--libnvvm-manifest",
            tampered.to_str().unwrap(),
            "--nvvm-check",
        ],
        &[],
        &dir,
    );
    assert_eq!(out.code, Some(2));
    assert!(
        out.stderr.contains("not the pinned libNVVM patch manifest"),
        "{}",
        out.stderr
    );
}

/// Tile IR -> libNVVM -> PTX -> native ptxas -> cubin, entirely in the helper
/// and its one admitted child. The bytecode comes from cuda-tile's pure-Python
/// writer in an UNPACKED wheel (build.sh fetches and verifies it).
#[test]
#[ignore = "executes NVIDIA code in the helper process and needs build.sh's inputs (cuda-tile wheel, native ptxas): set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn vector_add_tile_ir_compiles_to_an_sm_120_cubin() {
    let dir = scratch("compile");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::remove_file(path).unwrap();
        }
    }
    let wheel = std::env::var("TILEIRAS_NATIVE_CUDA_TILE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_file("target/pip/unpacked"));
    assert!(
        wheel.join("cuda/tile/_bytecode/writer.py").is_file(),
        "unpacked cuda-tile wheel missing at {} (run build.sh)",
        wheel.display()
    );
    let resolver = repo_file("../../tools/ptxas-resolve.sh");
    let elf = pinned_elf();
    for kernel in ["add_f32", "vecadd_f32"] {
        let input = format!("{kernel}.tilebc");
        let made = Command::new("python3")
            .args([
                "-I",
                repo_file("scripts/make_tileir_fixture.py")
                    .to_str()
                    .unwrap(),
                "--cuda-tile-root",
                wheel.to_str().unwrap(),
                "--kernel",
                kernel,
                "--out",
            ])
            .arg(dir.join(&input))
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            made.status.success(),
            "{}",
            String::from_utf8_lossy(&made.stderr)
        );
        let (cubin, ptx) = (format!("{kernel}.cubin"), dir.join(format!("{kernel}.ptx")));
        let out = helper(
            &[
                "--elf",
                &elf,
                "--timeout",
                "45",
                "--ptxas-resolver",
                resolver.to_str().unwrap(),
                "--",
                "--gpu-name=sm_120",
                &input,
                "-o",
                &cubin,
            ],
            &[("TILEIRAS_NATIVE_KEEP_PTX", ptx.to_str().unwrap())],
            &dir,
        );
        assert_eq!(out.code, Some(0), "{kernel}: {}", out.stderr);
        let ptx = std::fs::read_to_string(&ptx).unwrap();
        assert!(
            ptx.contains("Generated by NVIDIA NVVM Compiler") && ptx.contains(".target sm_120a"),
            "{ptx}"
        );
        assert!(ptx.contains(&format!(".visible .entry {kernel}(")), "{ptx}");
        // Structural validation of the cubin: ELF64 LE, EM_CUDA, executable,
        // SM 120 in e_flags, and the kernel's code + info sections.
        let bytes = std::fs::read(dir.join(&cubin)).unwrap();
        assert_eq!(&bytes[..6], b"\x7fELF\x02\x01");
        assert_eq!(u16::from_le_bytes([bytes[16], bytes[17]]), 2, "ET_EXEC");
        assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), 190, "EM_CUDA");
        assert_eq!(
            (u32::from_le_bytes(bytes[48..52].try_into().unwrap()) >> 8) & 0xff,
            120,
            "SM version in e_flags"
        );
        let sections = tileiras_native::rebase_audit::sections(&bytes).unwrap();
        let size_of = |name: &str| sections.iter().find(|s| s.name == name).map(|s| s.size);
        assert!(
            size_of(&format!(".text.{kernel}")).is_some_and(|n| n >= 64 && n % 16 == 0),
            "{sections:?}"
        );
        assert!(size_of(&format!(".nv.info.{kernel}")).is_some_and(|n| n > 0));
        assert!(size_of(".nv.info").is_some_and(|n| n > 0));
        assert!(size_of(&format!(".nv.constant0.{kernel}")).is_some());
    }
    // Temporary files stayed in the scratch directory and were removed by tileiras.
    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(
        left,
        [
            ".tileiras-native-bin",
            "add_f32.cubin",
            "add_f32.ptx",
            "add_f32.tilebc",
            "vecadd_f32.cubin",
            "vecadd_f32.ptx",
            "vecadd_f32.tilebc"
        ]
    );
}

/// Determinism and attribution. Two compiles in fresh directories give
/// byte-identical cubins; the PTX that tileiras + libNVVM hand to ptxas equals
/// the recorded tests/fixtures/vecadd_f32.expected.ptx.txt; and assembling that PTX
/// with the native ptxas directly (no tileiras) gives the same kernel code and
/// constants. So the loader-run compiler produced that PTX, and ptxas, not the
/// loader, produced the machine code.
#[test]
#[ignore = "executes NVIDIA code in the helper process and needs build.sh's inputs (cuda-tile wheel, native ptxas): set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn vector_add_compile_is_deterministic_and_its_ptx_assembles_to_the_same_code() {
    let wheel = std::env::var("TILEIRAS_NATIVE_CUDA_TILE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_file("target/pip/unpacked"));
    let resolver = repo_file("../../tools/ptxas-resolve.sh");
    let elf = pinned_elf();
    let mut runs = Vec::new();
    for name in ["determinism-a", "determinism-b"] {
        let dir = scratch(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let made = Command::new("python3")
            .args([
                "-I",
                repo_file("scripts/make_tileir_fixture.py")
                    .to_str()
                    .unwrap(),
                "--cuda-tile-root",
                wheel.to_str().unwrap(),
                "--kernel",
                "vecadd_f32",
                "--out",
            ])
            .arg(dir.join("vecadd_f32.tilebc"))
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            made.status.success(),
            "{}",
            String::from_utf8_lossy(&made.stderr)
        );
        let ptx = dir.join("vecadd_f32.ptx");
        let out = helper(
            &[
                "--elf",
                &elf,
                "--timeout",
                "45",
                "--ptxas-resolver",
                resolver.to_str().unwrap(),
                "--",
                "--gpu-name=sm_120",
                "vecadd_f32.tilebc",
                "-o",
                "vecadd_f32.cubin",
            ],
            &[("TILEIRAS_NATIVE_KEEP_PTX", ptx.to_str().unwrap())],
            &dir,
        );
        assert_eq!(out.code, Some(0), "{}", out.stderr);
        runs.push((
            std::fs::read(dir.join("vecadd_f32.tilebc")).unwrap(),
            std::fs::read(&ptx).unwrap(),
            std::fs::read(dir.join("vecadd_f32.cubin")).unwrap(),
        ));
    }
    assert_eq!(runs[0].0, runs[1].0, "the bytecode writer is deterministic");
    assert_eq!(runs[0].1, runs[1].1, "PTX differs between two runs");
    assert_eq!(runs[0].2, runs[1].2, "cubin differs between two runs");
    let expected = std::fs::read(repo_file("tests/fixtures/vecadd_f32.expected.ptx.txt")).unwrap();
    assert!(
        runs[0].1 == expected,
        "PTX from tileiras differs from tests/fixtures/vecadd_f32.expected.ptx.txt"
    );

    // The same PTX through the native ptxas alone (tileiras also passes
    // --nv-host=<its temporary object>, which only adds the .nv.host sections).
    let found = Command::new(&resolver)
        .arg("--print-path")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        found.status.success(),
        "{}",
        String::from_utf8_lossy(&found.stderr)
    );
    let ptxas = String::from_utf8(found.stdout).unwrap().trim().to_string();
    let dir = scratch("determinism-direct");
    let mut direct = Command::new(&ptxas);
    direct
        .args(["-arch", "sm_120a"])
        .arg(repo_file("tests/fixtures/vecadd_f32.expected.ptx.txt"))
        .args([
            "-o",
            "direct.cubin",
            "--opt-level=3",
            "--enable-extended-smem=true",
            "-uumn",
        ])
        .current_dir(&dir);
    let direct = bounded(direct, "native ptxas on the recorded PTX");
    assert_eq!(direct.code, Some(0), "{}", direct.stderr);
    let direct = std::fs::read(dir.join("direct.cubin")).unwrap();
    let section = |bytes: &[u8], name: &str| -> Vec<u8> {
        let s = tileiras_native::rebase_audit::sections(bytes)
            .unwrap()
            .into_iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{name} missing"));
        bytes[s.offset as usize..(s.offset + s.size) as usize].to_vec()
    };
    for name in [
        ".text.vecadd_f32",
        ".nv.constant0.vecadd_f32",
        ".nv.merc.nv.info.vecadd_f32",
        ".nv.callgraph",
    ] {
        assert!(
            section(&runs[0].2, name) == section(&direct, name),
            "{name} differs between the tileiras run and ptxas alone"
        );
    }
    assert_eq!(section(&direct, ".text.vecadd_f32").len(), 1024);
    let names = |bytes: &[u8]| {
        tileiras_native::rebase_audit::sections(bytes)
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect::<Vec<_>>()
    };
    let only_with_tileiras: Vec<String> = names(&runs[0].2)
        .into_iter()
        .filter(|n| !names(&direct).contains(n))
        .collect();
    assert_eq!(only_with_tileiras, [".nv.host", ".nv.merc.nv.host"]);
}

/// The child-process shim never falls back: without a verified native ptxas
/// the compile stops with a named fatal error and no cubin.
#[test]
#[ignore = "executes NVIDIA code in the helper process and needs build.sh's inputs: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn compile_fails_loudly_when_the_resolver_selects_no_ptxas() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("no-ptxas");
    let wheel = std::env::var("TILEIRAS_NATIVE_CUDA_TILE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_file("target/pip/unpacked"));
    let made = Command::new("python3")
        .args([
            "-I",
            repo_file("scripts/make_tileir_fixture.py")
                .to_str()
                .unwrap(),
            "--cuda-tile-root",
            wheel.to_str().unwrap(),
            "--out",
        ])
        .arg(dir.join("add_f32.tilebc"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    let resolver = dir.join("resolver-exit-3.sh");
    std::fs::write(
        &resolver,
        "#!/bin/sh\necho 'no verified native ptxas (test double)' >&2\nexit 3\n",
    )
    .unwrap();
    std::fs::set_permissions(&resolver, std::fs::Permissions::from_mode(0o755)).unwrap();
    let elf = pinned_elf();
    for resolver_args in [
        &["--ptxas-resolver", resolver.to_str().unwrap()][..],
        &[][..],
    ] {
        let mut args = vec!["--elf", &elf, "--timeout", "45"];
        args.extend_from_slice(resolver_args);
        args.extend_from_slice(&[
            "--",
            "--gpu-name=sm_120",
            "add_f32.tilebc",
            "-o",
            "out.cubin",
        ]);
        let out = helper(&args, &[("PATH", "/usr/bin:/bin")], &dir);
        if resolver_args.is_empty() {
            // No resolver, no `ptxas` in PATH: NVIDIA's own "not found" error.
            assert_ne!(out.code, Some(0), "{}", out.stderr);
        } else {
            assert_eq!(out.code, Some(70), "{}", out.stderr);
            assert!(
                out.stderr.contains("cannot run ptxas for the guest")
                    && out.stderr.contains("selected no verified ptxas"),
                "{}",
                out.stderr
            );
        }
        assert!(
            !dir.join("out.cubin").exists(),
            "no cubin may exist when ptxas did not run"
        );
    }

    // A relative resolver path names the caller's file, not a file of the
    // same name in the scratch directory the helper changes into.
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::copy(dir.join("add_f32.tilebc"), sub.join("add_f32.tilebc")).unwrap();
    let decoy = sub.join("resolver-exit-3.sh");
    std::fs::write(&decoy, "#!/bin/sh\ntouch decoy-ran\necho /bin/echo\n").unwrap();
    std::fs::set_permissions(&decoy, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = helper(
        &[
            "--elf",
            &elf,
            "--timeout",
            "45",
            "--scratch",
            "sub",
            "--ptxas-resolver",
            "resolver-exit-3.sh",
            "--",
            "--gpu-name=sm_120",
            "add_f32.tilebc",
            "-o",
            "out.cubin",
        ],
        &[("PATH", "/usr/bin:/bin")],
        &dir,
    );
    assert_eq!(out.code, Some(70), "{}", out.stderr);
    assert!(
        out.stderr
            .contains("no verified native ptxas (test double)"),
        "{}",
        out.stderr
    );
    assert!(
        !sub.join("decoy-ran").exists(),
        "the scratch directory's file of the same name must not run"
    );
    let missing = helper(
        &[
            "--elf",
            &elf,
            "--ptxas-resolver",
            "no-such-resolver.sh",
            "--",
            "--version",
        ],
        &[],
        &dir,
    );
    assert_eq!(missing.code, Some(2));

    // A resolver that selects a file outside a content-addressed directory, or
    // whose content does not match its directory name, is refused before exec.
    let rogue = dir.join("resolver-rogue.sh");
    std::fs::write(&rogue, "#!/bin/sh\necho /bin/echo\n").unwrap();
    std::fs::set_permissions(&rogue, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = helper(
        &[
            "--elf",
            &elf,
            "--timeout",
            "45",
            "--ptxas-resolver",
            rogue.to_str().unwrap(),
            "--",
            "--gpu-name=sm_120",
            "add_f32.tilebc",
            "-o",
            "out.cubin",
        ],
        &[("PATH", "/usr/bin:/bin")],
        &dir,
    );
    assert_eq!(out.code, Some(70), "{}", out.stderr);
    assert!(
        out.stderr
            .contains("/bin/echo is not the pinned compiler's content-addressed copy"),
        "{}",
        out.stderr
    );
    assert!(!dir.join("out.cubin").exists());

    // The PATH placeholder is never written through a symbolic link.
    let trap = scratch("placeholder-symlink");
    let _ = std::fs::remove_dir_all(&trap);
    std::fs::create_dir_all(trap.join(".tileiras-native-bin")).unwrap();
    std::fs::write(trap.join("victim"), "untouched").unwrap();
    std::os::unix::fs::symlink(trap.join("victim"), trap.join(".tileiras-native-bin/ptxas"))
        .unwrap();
    let out = helper(
        &[
            "--elf",
            &elf,
            "--ptxas-resolver",
            resolver.to_str().unwrap(),
            "--",
            "--version",
        ],
        &[],
        &trap,
    );
    assert_eq!(out.code, Some(2), "{}", out.stderr);
    assert!(
        out.stderr.contains("is not the helper's placeholder"),
        "{}",
        out.stderr
    );
    assert_eq!(
        std::fs::read_to_string(trap.join("victim")).unwrap(),
        "untouched"
    );
    assert!(
        out.stdout.is_empty(),
        "no guest code may run after the refusal"
    );
}

#[test]
#[ignore = "executes NVIDIA code in the helper process: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn constructors_run_once_and_the_real_version_banner_is_printed() {
    let dir = scratch("version");
    let elf = pinned_elf();
    let ctors = helper(
        &["--elf", &elf, "--", "--version"],
        &[("TILEIRAS_NATIVE_STOP_AFTER_CTORS", "1")],
        &dir,
    );
    assert_eq!(ctors.code, Some(0), "{}", ctors.stderr);
    assert!(ctors.stderr.contains("constructors completed"));
    assert!(ctors.stdout.is_empty());

    let version = guest(&["--version"], &dir);
    assert_eq!(version.code, Some(0), "{}", version.stderr);
    assert!(version
        .stdout
        .starts_with("tileiras: NVIDIA (R) Cuda Tile IR optimizing assembler\n"));
    assert!(version
        .stdout
        .contains("Cuda compilation tools, release 13.2, V13.2.51\n"));
    assert!(version.stderr.is_empty(), "{}", version.stderr);
}

#[test]
#[ignore = "executes NVIDIA code in the helper process: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn help_lists_the_real_options() {
    let help = guest(&["--help"], &scratch("help"));
    assert_eq!(help.code, Some(0), "{}", help.stderr);
    for needle in [
        "USAGE: tileiras [options] <tile bytecode file>",
        "--gpu-name=<value>",
        "=sm_120",
        "--host-arch=<value>",
        "--opt-level=<N>",
    ] {
        assert!(help.stdout.contains(needle), "missing {needle}");
    }
}

#[test]
#[ignore = "executes NVIDIA code in the helper process: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn input_errors_come_from_the_compiler_with_translated_errno() {
    let dir = scratch("errors");
    let missing = guest(&["missing.tilebc"], &dir);
    assert_eq!(missing.code, Some(1));
    // ENOENT travelled Darwin errno -> Linux errno cell -> strerror shim.
    assert!(
        missing
            .stderr
            .contains("unable to read input file 'missing.tilebc': No such file or directory"),
        "{}",
        missing.stderr
    );
    std::fs::write(dir.join("bad.tilebc"), b"not tile ir").unwrap();
    let bad = guest(
        &["--gpu-name=sm_120", "bad.tilebc", "-o", "out.cubin"],
        &dir,
    );
    assert_eq!(bad.code, Some(3));
    assert!(
        bad.stderr
            .contains("input does not correspond to Tile IR bytecode"),
        "{}",
        bad.stderr
    );
    assert!(
        !dir.join("out.cubin").exists(),
        "no cubin may be produced from an invalid input"
    );
}

#[test]
#[ignore = "requires the pinned ELF and patch manifest: set TILEIRAS_NATIVE_TEST_ELF, use --include-ignored"]
fn a_modified_patch_manifest_is_refused() {
    let dir = scratch("manifest");
    let elf = pinned_elf();
    let home = std::env::var("HOME").unwrap();
    let real = std::env::var("TILEIRAS_NATIVE_MANIFEST")
        .unwrap_or_else(|_| format!("{home}/.cache/tileiras-native/13.2.51/patches.tsv"));
    let mut bytes = std::fs::read(real).unwrap();
    let last = bytes.len() - 2;
    bytes[last] ^= 1;
    let tampered = dir.join("patches.tsv");
    std::fs::write(&tampered, bytes).unwrap();
    for mode in [&["--map-check"][..], &["--", "--version"][..]] {
        let mut args = vec!["--elf", &elf, "--manifest", tampered.to_str().unwrap()];
        args.extend_from_slice(mode);
        let out = helper(&args, &[], &dir);
        assert_eq!(out.code, Some(2));
        assert!(
            out.stderr.contains("not the pinned patch manifest"),
            "{}",
            out.stderr
        );
        assert!(out.stdout.is_empty());
    }
}
