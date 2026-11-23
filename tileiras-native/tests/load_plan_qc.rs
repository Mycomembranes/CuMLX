// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Independent, non-executing ELF planner tests. No foreign entry points,
//! executable mappings, TLS registers, sockets, or GPU access. CLI subprocess
//! tests run only this crate's non-executing planner, never the ELF input.

use tileiras_native::plan;

const BASE: u64 = 0x400000;
const RW: u64 = BASE + 0x8000;
const PHOFF: usize = 64;
const SHOFF: usize = 0x1000;
const SHENT: usize = 64;
const TLS_PH: usize = 4;

fn u16_at(b: &mut [u8], off: usize, value: u16) {
    b[off..off + 2].copy_from_slice(&value.to_le_bytes());
}
fn u32_at(b: &mut [u8], off: usize, value: u32) {
    b[off..off + 4].copy_from_slice(&value.to_le_bytes());
}
fn u64_at(b: &mut [u8], off: usize, value: u64) {
    b[off..off + 8].copy_from_slice(&value.to_le_bytes());
}

fn program(b: &mut [u8], index: usize, row: [u64; 8]) {
    let off = PHOFF + index * 56;
    u32_at(b, off, row[0] as u32);
    u32_at(b, off + 4, row[1] as u32);
    for (i, value) in row[2..].iter().enumerate() {
        u64_at(b, off + 8 + i * 8, *value);
    }
}

fn fixture() -> Vec<u8> {
    let mut b = vec![0u8; 0x5000];
    b[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\x03");
    u16_at(&mut b, 16, 2);
    u16_at(&mut b, 18, 183);
    u32_at(&mut b, 20, 1);
    u64_at(&mut b, 24, BASE + 0x600);
    u64_at(&mut b, 32, PHOFF as u64);
    u64_at(&mut b, 40, SHOFF as u64);
    u16_at(&mut b, 52, 64);
    u16_at(&mut b, 54, 56);
    u16_at(&mut b, 56, 6);
    u16_at(&mut b, 58, 64);
    u16_at(&mut b, 60, 10);
    u16_at(&mut b, 62, 1);
    let interpreter = b"/lib/ld-linux-aarch64.so.1\0";
    b[0x300..0x300 + interpreter.len()].copy_from_slice(interpreter);
    program(&mut b, 0, [1, 5, 0, BASE, 0, 0x4000, 0x4000, 0x4000]);
    program(&mut b, 1, [1, 6, 0x4000, RW, 0, 0x1000, 0x2000, 0x4000]);
    program(
        &mut b,
        2,
        [
            3,
            4,
            0x300,
            BASE + 0x300,
            0,
            interpreter.len() as u64,
            interpreter.len() as u64,
            1,
        ],
    );
    program(&mut b, 3, [2, 6, 0x4100, RW + 0x100, 0, 256, 256, 8]);
    // Empty TLS template aliases the constructor VA, just as real tileiras does.
    program(&mut b, TLS_PH, [7, 4, 0x4500, RW + 0x500, 0, 0, 32, 8]);
    program(&mut b, 5, [6, 4, 64, BASE + 64, 0, 6 * 56, 6 * 56, 8]);
    u32_at(&mut b, 0x600, 0xd65f03c0);
    u32_at(&mut b, 0x604, 0xd65f03c0);
    let strings = b"\0malloc\0unported_data\0libc.so.6\0";
    b[0x4200..0x4200 + strings.len()].copy_from_slice(strings);
    u32_at(&mut b, 0x4300 + 24, 1);
    b[0x4300 + 24 + 4] = 0x12;
    u32_at(&mut b, 0x4300 + 48, 8);
    b[0x4300 + 48 + 4] = 0x11;
    u64_at(&mut b, 0x4400, RW + 0x600);
    u64_at(&mut b, 0x4408, (1 << 32) | 1025);
    u64_at(&mut b, 0x4418, RW + 0x608);
    u64_at(&mut b, 0x4420, (2 << 32) | 1025);
    u64_at(&mut b, 0x4500, BASE + 0x600);
    u64_at(&mut b, 0x4508, BASE + 0x604);
    let tags = [
        (1, 22),
        (5, RW + 0x200),
        (10, strings.len() as u64),
        (6, RW + 0x300),
        (11, 24),
        (7, RW + 0x400),
        (8, 48),
        (9, 24),
        (12, BASE + 0x600),
        (13, BASE + 0x604),
        (25, RW + 0x500),
        (27, 8),
        (26, RW + 0x508),
        (28, 8),
        (0, 0),
    ];
    for (i, (tag, value)) in tags.iter().enumerate() {
        u64_at(&mut b, 0x4100 + i * 16, *tag);
        u64_at(&mut b, 0x4108 + i * 16, *value);
    }
    let names = b"\0.shstrtab\0.dynstr\0.dynsym\0.rela.dyn\0.init_array\0.fini_array\0.text\0.data\0.tbss\0";
    b[0x800..0x800 + names.len()].copy_from_slice(names);
    let rows = [
        (
            ".shstrtab",
            3,
            0,
            BASE + 0x800,
            0x800,
            names.len() as u64,
            0,
            1,
            0,
        ),
        (
            ".dynstr",
            3,
            2,
            RW + 0x200,
            0x4200,
            strings.len() as u64,
            0,
            1,
            0,
        ),
        (".dynsym", 11, 2, RW + 0x300, 0x4300, 72, 2, 8, 24),
        (".rela.dyn", 4, 2, RW + 0x400, 0x4400, 48, 3, 8, 24),
        (".init_array", 14, 3, RW + 0x500, 0x4500, 8, 0, 8, 8),
        (".fini_array", 15, 3, RW + 0x508, 0x4508, 8, 0, 8, 8),
        (".text", 1, 6, BASE + 0x600, 0x600, 8, 0, 4, 0),
        (".data", 1, 3, RW + 0x600, 0x4600, 16, 0, 8, 0),
        (".tbss", 8, 0x403, RW + 0x500, 0x4500, 32, 0, 8, 0),
    ];
    for (i, (name, kind, flags, va, off, size, link, align, entsize)) in rows.iter().enumerate() {
        let s = SHOFF + (i + 1) * SHENT;
        let name_offset = names
            .windows(name.len())
            .position(|v| v == name.as_bytes())
            .unwrap();
        u32_at(&mut b, s, name_offset as u32);
        u32_at(&mut b, s + 4, *kind);
        u64_at(&mut b, s + 8, *flags);
        u64_at(&mut b, s + 16, *va);
        u64_at(&mut b, s + 24, *off);
        u64_at(&mut b, s + 32, *size);
        u32_at(&mut b, s + 40, *link);
        u64_at(&mut b, s + 48, *align);
        u64_at(&mut b, s + 56, *entsize);
    }
    b
}

#[test]
fn independent_plan_keeps_imports_and_blocks_execution() {
    let b = fixture();
    let p = plan(&b, 16384).unwrap();
    assert_eq!(p.entry, BASE + 0x600);
    assert_eq!(p.load_base, BASE);
    assert_eq!(p.load_end, RW + 0x4000);
    assert_eq!(p.imports.len(), 2);
    assert_eq!(p.relocations.len(), 2);
    assert_eq!(p.needed_libraries, ["libc.so.6"]);
    assert_eq!(p.startup.init_array, [BASE + 0x600]);
    assert_eq!(p.startup.fini_array, [BASE + 0x604]);
    assert!(!p.execution_ready());
}

#[test]
fn tls_alias_does_not_zero_constructors_or_share_thread_storage() {
    let b = fixture();
    let original = b.clone();
    let p = plan(&b, 16384).unwrap();
    let image = p.materialize(&b).unwrap();
    let template = p.tls.as_ref().unwrap();
    assert_eq!(template.virtual_address, RW + 0x500);
    assert!(template.initialized_bytes.is_empty());
    let mut first = template.instantiate().unwrap();
    let second = template.instantiate().unwrap();
    assert_eq!(first, vec![0; 32]);
    first[0] = 77;
    assert_eq!(second, vec![0; 32]);
    assert_eq!(&image[0x8500..0x8508], &(BASE + 0x600).to_le_bytes());
    assert_eq!(&image[0x8508..0x8510], &(BASE + 0x604).to_le_bytes());
    assert!(image[0x9000..0xa000].iter().all(|v| *v == 0));
    assert_eq!(b, original);
    assert!(!p.execution_ready());
}

#[test]
fn initialized_tls_copies_template_and_zero_fills_only_private_tail() {
    let mut b = fixture();
    u64_at(&mut b, PHOFF + TLS_PH * 56 + 32, 8);
    let p = plan(&b, 16384).unwrap();
    let tls = p.tls.as_ref().unwrap();
    assert_eq!(tls.initialized_bytes, (BASE + 0x600).to_le_bytes());
    let copy = tls.instantiate().unwrap();
    assert_eq!(&copy[..8], &(BASE + 0x600).to_le_bytes());
    assert!(copy[8..].iter().all(|v| *v == 0));
    assert_eq!(b[0x4500..0x4508], (BASE + 0x600).to_le_bytes());
}

#[test]
fn initialized_tls_must_match_its_file_backed_load_mapping() {
    let mut b = fixture();
    u64_at(&mut b, PHOFF + TLS_PH * 56 + 32, 8);
    u64_at(&mut b, PHOFF + TLS_PH * 56 + 8, 0x4508);
    assert!(plan(&b, 16384).is_err());
}

#[test]
fn duplicate_tls_segments_are_rejected() {
    let mut b = fixture();
    let copied = b[PHOFF + TLS_PH * 56..PHOFF + (TLS_PH + 1) * 56].to_vec();
    u16_at(&mut b, 56, 7);
    u64_at(&mut b, PHOFF + 5 * 56 + 32, 7 * 56);
    u64_at(&mut b, PHOFF + 5 * 56 + 40, 7 * 56);
    b[PHOFF + 6 * 56..PHOFF + 7 * 56].copy_from_slice(&copied);
    assert!(plan(&b, 16384).is_err());
}

#[test]
fn contradictory_or_duplicate_explicit_phdr_is_rejected() {
    for (field, value) in [(8, 65), (16, BASE + 68), (32, 6 * 56 - 1), (40, 6 * 56 + 1)] {
        let mut b = fixture();
        u64_at(&mut b, PHOFF + 5 * 56 + field, value);
        assert!(
            plan(&b, 16384).is_err(),
            "accepted contradictory PT_PHDR field {field}"
        );
    }
    let mut b = fixture();
    u16_at(&mut b, 56, 7);
    u64_at(&mut b, PHOFF + 5 * 56 + 32, 7 * 56);
    u64_at(&mut b, PHOFF + 5 * 56 + 40, 7 * 56);
    let copied = b[PHOFF + 5 * 56..PHOFF + 6 * 56].to_vec();
    b[PHOFF + 6 * 56..PHOFF + 7 * 56].copy_from_slice(&copied);
    assert!(plan(&b, 16384).is_err());
}

#[test]
fn relro_subpage_metadata_does_not_promise_exact_protection() {
    let mut b = fixture();
    program(
        &mut b,
        5,
        [0x6474e552, 4, 0x4100, RW + 0x100, 0, 0x700, 0x700, 1],
    );
    let p = plan(&b, 16384).unwrap();
    assert_eq!(p.relro.len(), 1);
    let r = &p.relro[0];
    assert_eq!((r.virtual_address, r.size), (RW + 0x100, 0x700));
    assert_eq!((r.page_start, r.page_end), (RW, RW + 0x4000));
    assert_eq!(r.whole_page_start, r.whole_page_end);
    assert!(!r.exact_page_protection);
    assert!(!p.execution_ready());
    u64_at(&mut b, PHOFF + 5 * 56 + 8, 0x4108);
    assert!(plan(&b, 16384).is_err());
}

#[test]
fn empty_tls_file_offset_is_not_mistaken_for_template_bytes() {
    let mut b = fixture();
    u64_at(&mut b, PHOFF + TLS_PH * 56 + 8, 0x300);
    let p = plan(&b, 16384).unwrap();
    assert!(p.tls.as_ref().unwrap().initialized_bytes.is_empty());
    assert_eq!(p.tls.as_ref().unwrap().instantiate().unwrap(), vec![0; 32]);
}

#[test]
fn page_rounding_preserves_the_non_writable_text_separation() {
    let p = plan(&fixture(), 16384).unwrap();
    assert_eq!(p.host_page_size, 16384);
    assert_eq!(p.segments.len(), 2);
    assert_eq!(
        (p.segments[0].page_start, p.segments[0].page_end),
        (BASE, BASE + 0x4000)
    );
    assert_eq!(
        (p.segments[1].page_start, p.segments[1].page_end),
        (RW, RW + 0x4000)
    );
    assert!(p.segments[0].execute && !p.segments[0].write);
    assert!(p.segments[1].write && !p.segments[1].execute);
}

#[test]
fn byte_disjoint_segments_sharing_a_host_page_are_rejected() {
    let mut b = fixture();
    u64_at(&mut b, PHOFF + 32, 0x3000);
    u64_at(&mut b, PHOFF + 40, 0x3000);
    // Reuse the optional PT_PHDR slot for a third, logically disjoint load.
    // It is valid at 4-KiB page granularity but shares an RX page at 16 KiB.
    program(
        &mut b,
        5,
        [1, 6, 0x3000, BASE + 0x3000, 0, 0x1000, 0x1000, 0x1000],
    );
    assert!(plan(&b, 4096).is_ok());
    assert!(plan(&b, 16384).is_err());
}

#[test]
fn materialization_refuses_mutated_input_and_mutated_plan() {
    let b = fixture();
    let p = plan(&b, 16384).unwrap();
    let mut changed = b.clone();
    u64_at(&mut changed, 0x4500, BASE + 0x604);
    // Both images are individually valid; a stale admission must not mix them.
    assert!(plan(&changed, 16384).is_ok());
    assert!(p.materialize(&changed).is_err());
    let mut tampered_plan = p.clone();
    tampered_plan.segments[0].write = true;
    assert!(tampered_plan.materialize(&b).is_err());
    let mut tampered_hash = p.clone();
    tampered_hash.input_sha256 = "0".repeat(64);
    assert!(tampered_hash.materialize(&b).is_err());
    let mut unbounded = p.clone();
    unbounded.load_end = u64::MAX;
    assert!(unbounded.materialize(&b).is_err());
    let mut underflow = p.clone();
    underflow.load_base = u64::MAX;
    assert!(underflow.materialize(&b).is_err());
}

#[test]
fn malformed_headers_extents_and_relocations_are_rejected() {
    let mut cases = Vec::new();
    for (offset, value) in [
        (32, u64::MAX - 12),
        (40, u64::MAX - 12),
        (PHOFF + 32, u64::MAX),
        (PHOFF + 16, u64::MAX - 1024),
        (PHOFF + 40, u64::MAX),
        (24, RW + 0x600),
        (0x4400, BASE - 8),
        (0x4400, RW + 0x2000 - 4),
        (0x4408, (99 << 32) | 1025),
        (0x4408, (1 << 32) | 1027),
        (PHOFF + TLS_PH * 56 + 16, RW + 0x3000),
    ] {
        let mut b = fixture();
        u64_at(&mut b, offset, value);
        cases.push((offset, value, b));
    }
    for (offset, value, b) in cases {
        assert!(
            plan(&b, 16384).is_err(),
            "accepted mutation {offset:#x}={value:#x}"
        );
    }
    for length in [0, 63, 256, 0x4fff] {
        assert!(
            plan(&fixture()[..length], 16384).is_err(),
            "accepted length {length}"
        );
    }
}

#[test]
fn bounded_header_table_mutations_never_panic() {
    let original = fixture();
    let mut seed = 0x57a1_1e18_65b0_1234u64;
    for index in 0..256 {
        let mut b = original.clone();
        for _ in 0..4 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let base = match index % 4 {
                0 => 0,
                1 => SHOFF,
                2 => 0x4100,
                _ => 0x4300,
            };
            let off = base + ((seed >> 8) as usize % 256);
            b[off] ^= seed as u8;
        }
        assert!(
            std::panic::catch_unwind(|| plan(&b, 16384)).is_ok(),
            "parser panicked for deterministic mutation {index}"
        );
    }
}

#[test]
fn constructor_pointer_must_be_executable_file_backed_code() {
    for target in [RW + 0x600, RW + 0x1800, BASE - 4, BASE + 0x7000] {
        let mut b = fixture();
        u64_at(&mut b, 0x4500, target);
        assert!(plan(&b, 16384).is_err(), "accepted ctor {target:#x}");
    }
}

#[test]
fn writable_executable_load_segment_is_rejected() {
    let mut b = fixture();
    u32_at(&mut b, PHOFF + 4, 7);
    assert!(plan(&b, 16384).is_err());
}

#[test]
fn malformed_page_sizes_are_rejected() {
    for size in [0, 1, 3, 4095, 16383, u64::MAX] {
        assert!(plan(&fixture(), size).is_err(), "accepted page size {size}");
    }
}

#[test]
#[ignore = "requires local pinned ELF: set TILEIRAS_NATIVE_TEST_ELF and use --include-ignored"]
fn real_pinned_tileiras_metadata_matches_independent_python_audit() {
    let path = std::env::var_os("TILEIRAS_NATIVE_TEST_ELF")
        .expect("set TILEIRAS_NATIVE_TEST_ELF to the separately SHA256-pinned local NVIDIA ELF");
    let b = std::fs::read(path).unwrap();
    assert_eq!(b.len(), 90_527_008);
    let p = plan(&b, 16384).unwrap();
    assert_eq!(
        p.input_sha256,
        "297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a"
    );
    assert_eq!(p.entry, 0x4fec10);
    assert_eq!(p.load_base, 0x400000);
    assert_eq!(p.imports.len(), 276);
    assert_eq!(p.relocations.len(), 294);
    assert_eq!(p.startup.init_array.len(), 678);
    assert_eq!(p.startup.fini_array.len(), 1);
    assert_eq!(p.relro.len(), 1);
    let relro = &p.relro[0];
    assert_eq!(relro.virtual_address, 0x4d74218);
    assert_eq!(relro.virtual_address + relro.size, 0x5942000);
    assert_eq!((relro.page_start, relro.page_end), (0x4d74000, 0x5944000));
    assert_eq!(
        (relro.whole_page_start, relro.whole_page_end),
        (0x4d78000, 0x5940000)
    );
    assert!(!relro.exact_page_protection);
    assert_eq!(
        (p.segments[0].page_start, p.segments[0].page_end),
        (0x400000, 0x4d64000)
    );
    assert_eq!(
        (p.segments[1].page_start, p.segments[1].page_end),
        (0x4d74000, 0x5b1c000)
    );
    let template = p.tls.as_ref().unwrap();
    assert_eq!(template.virtual_address, 0x5806868);
    assert_eq!(template.memory_size, 772);
    assert_eq!(template.alignment, 8);
    assert!(template.initialized_bytes.is_empty());
    let image = p.materialize(&b).unwrap();
    let offset = (template.virtual_address - p.load_base) as usize;
    let before = image[offset..offset + 678 * 8].to_vec();
    assert!(before.iter().any(|v| *v != 0));
    assert_eq!(template.instantiate().unwrap(), vec![0; 772]);
    assert_eq!(before, image[offset..offset + 678 * 8]);
    for (index, expected) in p.startup.init_array.iter().enumerate() {
        assert_eq!(
            u64::from_le_bytes(
                image[offset + index * 8..offset + index * 8 + 8]
                    .try_into()
                    .unwrap()
            ),
            *expected
        );
    }
    assert!(!p.execution_ready());
}

#[test]
fn cli_help_and_planner_version_work_without_an_elf_or_home() {
    for args in [
        vec![],
        vec!["-h"],
        vec!["--help"],
        vec!["plan", "--elf", "/nonexistent/tileiras-qc-input", "--help"],
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_tileiras-native"))
            .args(&args)
            .env_remove("HOME")
            .output()
            .unwrap();
        assert!(output.status.success(), "help failed: {args:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("NOT a runnable NVIDIA compiler"));
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_tileiras-native"))
        .arg("--version")
        .env_remove("HOME")
        .output()
        .unwrap();
    assert!(output.status.success());
    let version = String::from_utf8_lossy(&output.stdout);
    assert!(version.contains("planner 0.1.0"));
    assert!(version.contains("NVIDIA compiler execution NOT implemented"));
}

#[test]
fn cli_run_compile_and_bypass_flags_are_refused_before_input_access() {
    for args in [
        vec!["run", "--version"],
        vec!["compile", "input.tileir", "--gpu-name", "sm_120"],
        vec!["plan", "--allow-unpinned"],
        vec!["plan", "--execute"],
        vec!["plan", "--page-size", "8192"],
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_tileiras-native"))
            .args(&args)
            .env_remove("HOME")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "accepted {args:?}");
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("HOME missing"),
            "input accessed before refusal: {stderr}"
        );
    }
}

#[test]
fn cli_rejects_a_structurally_valid_unpinned_image() {
    use std::io::Write;
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("tileiras-qc-{}-{suffix}.elf", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let b = fixture();
    assert!(plan(&b, 16384).is_ok());
    file.write_all(&b).unwrap();
    drop(file);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_tileiras-native"))
        .args(["plan", "--elf"])
        .arg(&path)
        .arg("--materialize-check")
        .output();
    std::fs::remove_file(&path).unwrap();
    let output = output.unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unadmitted TileIR input SHA256"));
}
