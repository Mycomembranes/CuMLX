// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: `ptxas-darwin-arm64` — a ptxas-compatible CLI over
// NVIDIA's libnvptxcompiler_static.a (Linux-aarch64, CUDA 13.2.51) loaded
// in-process. Usage mirrors ptxas:
//   ptxas-darwin-arm64 --gpu-name=sm_120 -O3 in.ptx -o out.cubin
//   ptxas-darwin-arm64 --version | --self-test
//   ptxas-darwin-arm64 --identity                 # identity JSON; exit 0 verified, 4 not
//   ptxas-darwin-arm64 --manifest ... -o k.cubin  # also writes k.cubin.json
//                                                 # (or CUDA_OXIDE_PTXAS_MANIFEST=1)

// The loader only exists on macOS arm64; elsewhere the crate is a stub (ptxas there is real).
#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code, unused_imports)
)]

use std::process::ExitCode;

use ptxas_native::identity::{self, Identity};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: ptxas-darwin-arm64 [--gpu-name=sm_XX | -arch=sm_XX] [-O0..-O3] [-v] [ptxas options] in.ptx [-o out.cubin]\n\
             \x20      [--manifest]   also write <out>.json (ptx/cubin sha256, arch, options, compiler identity)\n\
             \x20      ptxas-darwin-arm64 --version | --identity | --self-test\n\
             env: PTXAS_NATIVE_ARCHIVE (default {}), PTXAS_NATIVE_X18_MANIFEST (default {}),\n\
             \x20    {}=1 (same as --manifest), PTXAS_NATIVE_TRACE=1",
            ptxas_native::default_archive().display(),
            ptxas_native::default_manifest().display(),
            identity::MANIFEST_ENV,
        );
        return ExitCode::from(if args.is_empty() { 2 } else { 0 });
    }
    if args.iter().any(|a| a == "--identity") {
        // Hash-only: never loads or runs the NVIDIA library.
        let id = Identity::current();
        println!("{}", id.to_json(""));
        return ExitCode::from(if id.verified() { 0 } else { 4 });
    }
    run(&args)
}

fn manifest_requested(flag: bool) -> bool {
    flag || std::env::var(identity::MANIFEST_ENV).is_ok_and(|v| v == "1")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run(args: &[String]) -> ExitCode {
    let cli = match ptxas_native::parse_cli(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ptxas-darwin-arm64: {e}");
            return ExitCode::from(2);
        }
    };
    let comp = match ptxas_native::NvPtxCompiler::load_default() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ptxas-darwin-arm64: load failed: {e}");
            return ExitCode::from(3);
        }
    };
    if cli.version || cli.self_test {
        let (maj, min) = comp.version().unwrap_or((0, 0));
        let id = &comp.identity;
        println!("ptxas-darwin-arm64: NVIDIA libnvptxcompiler_static {maj}.{min} (redist {}, linux-sbsa) loaded in-process on macOS arm64", ptxas_native::REDIST_VERSION);
        println!(
            "Cuda compilation tools, release {maj}.{min}, V{}",
            ptxas_native::REDIST_VERSION
        );
        println!(
            "wrapper {}  loader {}  archive sha256 {}  x18 manifest sha256 {}",
            identity::WRAPPER_VERSION,
            id.loader_git_describe.as_deref().unwrap_or("unknown"),
            id.archive_sha256.as_deref().unwrap_or("?"),
            id.x18_manifest_sha256.as_deref().unwrap_or("?")
        );
        if cli.self_test {
            let r = &comp.report;
            println!(
                "  members {}  image {} B (text {} B)  load {} ms",
                r.members, r.image_bytes, r.text_bytes, r.load_ms
            );
            println!(
                "  relocations {}  GOT {}  veneers {}  externals {}  COMDAT kept/discarded {}/{}  ctors {}",
                r.stats.relocations,
                r.stats.got_entries,
                r.stats.veneers,
                r.stats.externals,
                r.stats.comdat_groups_kept,
                r.stats.comdat_groups_discarded,
                r.stats.init_array_entries
            );
            println!(
                "  x18 renamed {}  islands {}  remat-defs {}  guarded (brk) functions {}",
                r.stats.x18_renamed,
                r.stats.x18_islands,
                r.stats.x18_remat_defs,
                r.stats.x18_trapped_functions
            );
            match comp.supported_archs() {
                Ok(a) => println!("  supported archs: {a:?}"),
                Err(e) => println!("  supported archs: error {e}"),
            }
        }
        if cli.input.is_none() {
            return ExitCode::SUCCESS;
        }
    }
    let Some(input) = cli.input else {
        eprintln!("ptxas-darwin-arm64: no input file");
        return ExitCode::from(2);
    };
    let ptx = match std::fs::read(&input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ptxas-darwin-arm64: {}: {e}", input.display());
            return ExitCode::from(2);
        }
    };
    let want_manifest = manifest_requested(cli.manifest);
    let out = match comp.compile(&ptx, &cli.options) {
        Ok(out) => out,
        Err(e) => {
            identity::unpublish(&cli.output);
            eprintln!("ptxas-darwin-arm64: {e}");
            return ExitCode::from(1);
        }
    };
    if !out.info_log.is_empty() {
        eprint!("{}", out.info_log);
    }
    // Publish only a validated cubin (type, machine, SM, entries, KPARAM vs PTX).
    let arch = identity::arch_of(&cli.options);
    let validated = match ptxas_native::validate::validate_cubin(&out.cubin, arch.as_deref(), &ptx)
    {
        Ok(v) => v,
        Err(e) => {
            identity::unpublish(&cli.output);
            eprintln!("ptxas-darwin-arm64: compiled cubin failed validation, not written: {e}");
            return ExitCode::from(1);
        }
    };
    let manifest = want_manifest.then(|| {
        identity::manifest_json(
            &input,
            &cli.output,
            &ptx,
            &out.cubin,
            &cli.options,
            &validated,
            &comp.identity,
        )
    });
    if let Err(e) = identity::publish(&cli.output, &out.cubin, manifest.as_deref()) {
        identity::unpublish(&cli.output);
        eprintln!("ptxas-darwin-arm64: write {}: {e}", cli.output.display());
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn run(_args: &[String]) -> ExitCode {
    eprintln!("ptxas-darwin-arm64 only runs on macOS arm64 (use the real ptxas elsewhere)");
    ExitCode::from(126)
}
