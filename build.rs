// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: embed `git describe` of the repo (loader + wrapper
// source) into ptxas-darwin-arm64's identity. Absent git -> no value (null).

use std::process::Command;

fn main() {
    let here = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../binary-ko-loader/src");
    if let Ok(o) = Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .current_dir(&here)
        .output()
    {
        if o.status.success() {
            let gd = String::from_utf8_lossy(&o.stdout).trim().to_string();
            let gd = std::path::Path::new(&here).join(gd);
            println!("cargo:rerun-if-changed={}", gd.join("HEAD").display());
            println!("cargo:rerun-if-changed={}", gd.join("index").display());
        }
    }
    let describe = Command::new("git")
        .args(["describe", "--always", "--dirty", "--abbrev=12"])
        .current_dir(&here)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(d) = describe {
        println!("cargo:rustc-env=PTXAS_NATIVE_GIT_DESCRIBE={d}");
    }
}
