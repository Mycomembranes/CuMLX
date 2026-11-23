// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: host-only end-to-end test of solution 105 Route B.
// Loads NVIDIA's Linux-aarch64 libnvptxcompiler_static.a in-process, compiles
// our vecadd PTX (fixtures/ptx) for sm_120 and requires every section except
// .note.nv.tkinfo to equal the Linux ptxas 13.2.78 reference cubin from the
// external reference bundle (SHA-checked). NO GPU.
// Missing prerequisites (redist archive/x18 manifest, or the reference bundle)
// print "SKIPPED: <reason>"; Rust cannot mark a test skipped, so set
// PTXAS_NATIVE_REQUIRE_FULL=1 (CI / pre-commit) to turn every skip into a failure.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::path::PathBuf;
use std::sync::OnceLock;

use ptxas_native::{default_archive, default_manifest, nv_info_attrs, parse_cubin, NvPtxCompiler};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/ptx")
        .join(name)
}

/// Print a loud skip; with PTXAS_NATIVE_REQUIRE_FULL=1 a skip is a failure.
fn skip(reason: &str) {
    if std::env::var("PTXAS_NATIVE_REQUIRE_FULL").is_ok_and(|v| v == "1") {
        panic!("required prerequisite missing: {reason}");
    }
    eprintln!("SKIPPED (not a pass): {reason}");
}

/// The SHA-checked reference cubin from the external bundle, or None (skip).
fn reference(rel: &str) -> Option<Vec<u8>> {
    let root = std::env::var_os("PTXAS_NATIVE_REF_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo().join("crates/cuda-driverkit/test_assets"));
    let list = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/reference-bundle.sha256"),
    )
    .expect("fixtures/reference-bundle.sha256");
    let want = list
        .lines()
        .find_map(|l| l.strip_suffix(rel).map(|h| h.trim().to_string()))
        .unwrap_or_else(|| panic!("{rel} not listed in reference-bundle.sha256"));
    let Ok(bytes) = std::fs::read(root.join(rel)) else {
        skip(&format!(
            "reference bundle file {} absent (see README 'Reference bundle')",
            root.join(rel).display()
        ));
        return None;
    };
    assert_eq!(
        ptxas_native::identity::sha256_hex(&bytes),
        want,
        "reference {rel} does not match reference-bundle.sha256"
    );
    Some(bytes)
}

fn compiler() -> Option<&'static NvPtxCompiler> {
    static C: OnceLock<Option<NvPtxCompiler>> = OnceLock::new();
    C.get_or_init(|| {
        if !default_archive().exists() || !default_manifest().exists() {
            skip("redist archive / x18 manifest absent (run portkit/ptxas-native/build.sh)");
            return None;
        }
        Some(NvPtxCompiler::load_default().expect("load libnvptxcompiler in-process"))
    })
    .as_ref()
}

#[test]
fn loads_and_reports_sm_120() {
    let Some(c) = compiler() else { return };
    assert_eq!(c.version().unwrap(), (13, 2));
    assert!(c.supported_archs().unwrap().contains(&120));
    let s = &c.report.stats;
    assert_eq!(c.report.members, 412);
    assert_eq!(s.init_array_entries, 7);
    assert!(
        s.x18_renamed > 2000 && s.x18_islands > 900,
        "x18 rewrite applied: {s:?}"
    );
}

#[test]
fn vecadd_text_matches_linux_ptxas() {
    let Some(c) = compiler() else { return };
    let Some(reference) = reference("cubins/vecadd_macos.cubin") else {
        return;
    };
    let ptx = std::fs::read(fixture("vecadd_macos.ptx")).unwrap();
    let out = c
        .compile(&ptx, &["--gpu-name=sm_120".to_string()])
        .expect("compile");
    let native = parse_cubin(&out.cubin).unwrap();
    let linux = parse_cubin(&reference).unwrap();
    assert_eq!(native.machine, ptxas_native::EM_CUDA);
    assert_eq!(native.sm, 120);
    for s in &linux.sections {
        if s.name == ".note.nv.tkinfo" {
            continue; // tool version string: 13.2.51 vs 13.2.78
        }
        let n = native
            .section(&s.name)
            .unwrap_or_else(|| panic!("missing {}", s.name));
        assert_eq!(n.data, s.data, "section {} differs", s.name);
    }
    let info = native.section(".nv.info.vecadd_macos").unwrap();
    let kparams = nv_info_attrs(&info.data)
        .iter()
        .filter(|a| a.1 == ptxas_native::EIATTR_KPARAM_INFO)
        .count();
    assert_eq!(kparams, 4);
}

#[test]
fn bad_ptx_reports_error_log_not_crash() {
    let Some(c) = compiler() else { return };
    let e = c
        .compile(
            b".version 8.7\n.target sm_120\n.address_size 64\n.visible .entry k( { bogus; }",
            &["--gpu-name=sm_120".into()],
        )
        .err()
        .expect("must fail");
    assert!(e.0.contains("NVPTXCOMPILE_ERROR"), "{e}");
}

#[test]
fn repeated_compiles_are_deterministic() {
    let Some(c) = compiler() else { return };
    let ptx = std::fs::read(fixture("vecadd_macos.ptx")).unwrap();
    let a = c
        .compile(&ptx, &["--gpu-name=sm_120".into(), "-O3".into()])
        .unwrap()
        .cubin;
    let b = c
        .compile(&ptx, &["--gpu-name=sm_120".into(), "-O3".into()])
        .unwrap()
        .cubin;
    assert_eq!(a, b);
}

#[test]
fn compiled_cubin_validates_and_mismatches_are_refused() {
    let Some(c) = compiler() else { return };
    let ptx = std::fs::read(fixture("vecadd_macos.ptx")).unwrap();
    let out = c.compile(&ptx, &["--gpu-name=sm_120".into()]).unwrap();
    let v = ptxas_native::validate::validate_cubin(&out.cubin, Some("sm_120"), &ptx).unwrap();
    assert_eq!(v.sm, 120);
    assert_eq!(v.kernels, vec!["vecadd_macos".to_string()]);
    // wrong requested SM
    assert!(ptxas_native::validate::validate_cubin(&out.cubin, Some("sm_90"), &ptx).is_err());
    // PTX that declares a different parameter list for the entry
    let other = String::from_utf8_lossy(&ptx).replacen(".param .u32", ".param .u64", 1);
    assert_ne!(
        other.as_bytes(),
        &ptx[..],
        "fixture has a .u32 param to mutate"
    );
    assert!(
        ptxas_native::validate::validate_cubin(&out.cubin, Some("sm_120"), other.as_bytes())
            .is_err()
    );
    // a truncated cubin is an error, not a panic
    assert!(ptxas_native::validate::validate_cubin(&out.cubin[..200], None, &ptx).is_err());
}
