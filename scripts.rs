// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: run the host-only script tests (cubin_diff gate,
// milestone failure on a changed .text byte, tools/ptxas-resolve.sh contract)
// as part of `cargo test`. Skips with a note when python3 is absent. Individual
// Python tests skip (with their reason, echoed here) when the reference bundle or
// the native build is absent; PTXAS_NATIVE_REQUIRE_FULL=1 turns skips into failures.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn gate_milestone_and_resolver_contract() {
    let test = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_gate_and_resolver.py");
    // Test the binary cargo just built, not a possibly stale target/release one.
    let out = match Command::new("python3")
        .arg("-I")
        .arg(&test)
        .env(
            "PTXAS_DARWIN_ARM64_BIN",
            env!("CARGO_BIN_EXE_ptxas-darwin-arm64"),
        )
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var("PTXAS_NATIVE_REQUIRE_FULL").as_deref() != Ok("1"),
                "python3 required: {e}"
            );
            eprintln!("SKIPPED (not a pass): python3 not runnable ({e})");
            return;
        }
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{stderr}");
    for l in stderr.lines().filter(|l| l.contains("skipped")) {
        eprintln!("SKIPPED (not a pass): {l}");
    }
    assert!(out.status.success(), "script tests failed:\n{stderr}");
}
