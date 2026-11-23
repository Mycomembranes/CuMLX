// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turns whatever a client hands to `cuModuleLoadData*` into an sm_120 cubin.
//!
//! * An ELF cubin is accepted only when it is already built for the card's SM.
//! * PTX text is compiled with the repo's verified native `ptxas`
//!   (`tools/ptxas-resolve.sh --print-path`), never a `ptxas` from `PATH`.
//! * A fat binary is searched for an uncompressed PTX entry or a matching ELF.
//!
//! Old applications that ship only pre-Blackwell cubins with no PTX cannot run;
//! that is reported as `CUDA_ERROR_NO_BINARY_FOR_GPU`.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{cu, CuError, CuResult};

/// SM version of the served card.
pub const TARGET_SM: u32 = 120;
/// Largest module the shared-CUDA backend admits.
pub const MAX_CUBIN_BYTES: usize = 8 * 1024 * 1024;

const ELF_MAGIC: &[u8; 4] = b"\x7fELF";
const FATBIN_MAGIC: u32 = 0xBA55_ED50;
const FATBIN_KIND_PTX: u16 = 1;
const FATBIN_KIND_ELF: u16 = 2;

/// JIT options carried by `cuModuleLoadDataEx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitOptions {
    /// `CU_JIT_OPTIMIZATION_LEVEL`, 0..=4 (4 is treated as 3 by ptxas).
    pub opt_level: u32,
}

impl Default for JitOptions {
    fn default() -> Self {
        Self { opt_level: 3 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    Cubin,
    Ptx,
    Fatbin,
}

/// Result of preparing a client image.
#[derive(Debug)]
pub struct Prepared {
    pub cubin: Vec<u8>,
    /// Compiler diagnostics (empty for a pass-through cubin).
    pub log: String,
    /// True when PTX was compiled here.
    pub compiled: bool,
}

pub fn classify(image: &[u8]) -> CuResult<ImageKind> {
    if image.len() >= 4 && &image[..4] == ELF_MAGIC {
        return Ok(ImageKind::Cubin);
    }
    if image.len() >= 4
        && u32::from_le_bytes([image[0], image[1], image[2], image[3]]) == FATBIN_MAGIC
    {
        return Ok(ImageKind::Fatbin);
    }
    if looks_like_ptx(image) {
        return Ok(ImageKind::Ptx);
    }
    Err(CuError::new(
        cu::INVALID_IMAGE,
        "module image is not an ELF cubin, PTX text or a fat binary",
    ))
}

fn looks_like_ptx(image: &[u8]) -> bool {
    let head = &image[..image.len().min(4096)];
    let Ok(text) = std::str::from_utf8(head.split(|b| *b == 0).next().unwrap_or(head)) else {
        return false;
    };
    text.contains(".version") && (text.contains(".target") || text.contains(".address_size"))
}

/// SM version from an ELF cubin's `e_flags` (bits 8..16).
pub fn cubin_sm(image: &[u8]) -> CuResult<u32> {
    // ELF64 header: e_flags is a little-endian u32 at offset 48.
    if image.len() < 64 || &image[..4] != ELF_MAGIC || image[4] != 2 || image[5] != 1 {
        return Err(CuError::new(
            cu::INVALID_IMAGE,
            "not a little-endian ELF64 cubin",
        ));
    }
    let flags = u32::from_le_bytes([image[48], image[49], image[50], image[51]]);
    Ok((flags >> 8) & 0xff)
}

pub fn prepare(image: &[u8], options: JitOptions, ptxas: &PtxasLocator) -> CuResult<Prepared> {
    let prepared = match classify(image)? {
        ImageKind::Cubin => pass_through_cubin(image)?,
        ImageKind::Ptx => compile_ptx(ptx_text(image), options, ptxas)?,
        ImageKind::Fatbin => prepare_fatbin(image, options, ptxas)?,
    };
    if prepared.cubin.len() > MAX_CUBIN_BYTES {
        return Err(CuError::new(
            cu::INVALID_IMAGE,
            format!(
                "cubin is {} bytes; the backend admits at most {MAX_CUBIN_BYTES}",
                prepared.cubin.len()
            ),
        ));
    }
    Ok(prepared)
}

fn pass_through_cubin(image: &[u8]) -> CuResult<Prepared> {
    let sm = cubin_sm(image)?;
    if sm != TARGET_SM {
        return Err(CuError::new(
            cu::NO_BINARY_FOR_GPU,
            format!("cubin targets sm_{sm}; this card needs sm_{TARGET_SM} (supply PTX so it can be recompiled)"),
        ));
    }
    Ok(Prepared {
        cubin: image.to_vec(),
        log: String::new(),
        compiled: false,
    })
}

/// PTX is a C string: stop at the first NUL if the client included one.
fn ptx_text(image: &[u8]) -> &[u8] {
    image.split(|b| *b == 0).next().unwrap_or(image)
}

/// How the verified ptxas is found.
#[derive(Debug, Clone)]
pub enum PtxasLocator {
    /// Run `<repo>/tools/ptxas-resolve.sh --print-path` (production).
    Resolver(PathBuf),
    /// Use this executable directly (tests and explicit operator override).
    Explicit(PathBuf),
}

impl PtxasLocator {
    /// `CUDA_ADAPTER_PTXAS` wins; otherwise the repo resolver under
    /// `CUDA_ADAPTER_REPO_ROOT` or the build-time repo root.
    pub fn from_env() -> Self {
        if let Some(path) = std::env::var_os("CUDA_ADAPTER_PTXAS").filter(|v| !v.is_empty()) {
            return Self::Explicit(PathBuf::from(path));
        }
        let root = std::env::var_os("CUDA_ADAPTER_REPO_ROOT")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."));
        Self::Resolver(root.join("tools/ptxas-resolve.sh"))
    }

    fn resolve(&self) -> CuResult<PathBuf> {
        match self {
            Self::Explicit(path) => Ok(path.clone()),
            Self::Resolver(script) => {
                let output = Command::new(script)
                    .arg("--print-path")
                    .output()
                    .map_err(|e| {
                        CuError::new(
                            cu::NO_BINARY_FOR_GPU,
                            format!("cannot run {}: {e}", script.display()),
                        )
                    })?;
                if !output.status.success() {
                    return Err(CuError::new(
                        cu::NO_BINARY_FOR_GPU,
                        format!(
                            "no verified ptxas on this host: {}",
                            String::from_utf8_lossy(&output.stderr).trim()
                        ),
                    ));
                }
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if path.is_empty() {
                    return Err(CuError::new(
                        cu::NO_BINARY_FOR_GPU,
                        "ptxas resolver printed no path",
                    ));
                }
                Ok(PathBuf::from(path))
            }
        }
    }
}

fn compile_ptx(ptx: &[u8], options: JitOptions, locator: &PtxasLocator) -> CuResult<Prepared> {
    let ptxas = locator.resolve()?;
    let dir = private_temp_dir()?;
    let result = run_ptxas(&ptxas, &dir, ptx, options);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn run_ptxas(ptxas: &Path, dir: &Path, ptx: &[u8], options: JitOptions) -> CuResult<Prepared> {
    let input = dir.join("module.ptx");
    let output = dir.join("module.cubin");
    std::fs::write(&input, ptx)
        .map_err(|e| CuError::new(cu::UNKNOWN, format!("cannot stage PTX: {e}")))?;
    let run = Command::new(ptxas)
        .arg(format!("--gpu-name=sm_{TARGET_SM}"))
        .arg(format!("-O{}", options.opt_level.min(3)))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .map_err(|e| {
            CuError::new(
                cu::NO_BINARY_FOR_GPU,
                format!("cannot run {}: {e}", ptxas.display()),
            )
        })?;
    let mut log = String::from_utf8_lossy(&run.stderr).into_owned();
    log.push_str(&String::from_utf8_lossy(&run.stdout));
    if !run.status.success() {
        return Err(CuError::new(
            cu::INVALID_IMAGE,
            format!("ptxas failed: {}", log.trim()),
        ));
    }
    let cubin = std::fs::read(&output)
        .map_err(|e| CuError::new(cu::INVALID_IMAGE, format!("ptxas produced no cubin: {e}")))?;
    let sm = cubin_sm(&cubin)?;
    if sm != TARGET_SM {
        return Err(CuError::new(
            cu::INVALID_IMAGE,
            format!("ptxas produced sm_{sm}, expected sm_{TARGET_SM}"),
        ));
    }
    Ok(Prepared {
        cubin,
        log,
        compiled: true,
    })
}

/// A fresh 0700 directory that only this user can read.
fn private_temp_dir() -> CuResult<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = format!(
        "cuda-adapter-ptx-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let dir = std::env::temp_dir().join(name);
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&dir)
        .map_err(|e| CuError::new(cu::UNKNOWN, format!("cannot create {}: {e}", dir.display())))?;
    Ok(dir)
}

// Fat binary layout (not published by NVIDIA; this follows the layout used by
// open-source decoders and is checked here only against a synthetic image):
//   header: u32 magic, u16 version, u16 header_size, u64 fat_size
//   entry : u16 kind, u16 version, u32 header_size, u64 size,
//           u32 compressed_size, u32 options_offset, u16 minor, u16 major,
//           u32 arch, u32 name_offset, u32 name_size, u64 flags, u64 zero,
//           u64 decompressed_size
// The payload of `size` bytes follows each entry header.
fn prepare_fatbin(image: &[u8], options: JitOptions, ptxas: &PtxasLocator) -> CuResult<Prepared> {
    let entries = fatbin_entries(image)?;
    if let Some(entry) = entries
        .iter()
        .find(|e| e.kind == FATBIN_KIND_ELF && e.arch == TARGET_SM && !e.compressed)
    {
        return pass_through_cubin(entry.payload);
    }
    if let Some(entry) = entries
        .iter()
        .find(|e| e.kind == FATBIN_KIND_PTX && !e.compressed)
    {
        return compile_ptx(ptx_text(entry.payload), options, ptxas);
    }
    let archs: Vec<String> = entries
        .iter()
        .map(|e| {
            format!(
                "{}:sm_{}{}",
                if e.kind == FATBIN_KIND_PTX {
                    "ptx"
                } else {
                    "elf"
                },
                e.arch,
                if e.compressed { "(compressed)" } else { "" }
            )
        })
        .collect();
    Err(CuError::new(
        cu::NO_BINARY_FOR_GPU,
        format!(
            "fat binary has no sm_{TARGET_SM} cubin and no uncompressed PTX (entries: {})",
            archs.join(", ")
        ),
    ))
}

#[derive(Debug)]
pub struct FatbinEntry<'a> {
    pub kind: u16,
    pub arch: u32,
    pub compressed: bool,
    pub payload: &'a [u8],
}

pub fn fatbin_entries(image: &[u8]) -> CuResult<Vec<FatbinEntry<'_>>> {
    let bad = |why: &str| CuError::new(cu::INVALID_IMAGE, format!("malformed fat binary: {why}"));
    if image.len() < 16 {
        return Err(bad("shorter than its header"));
    }
    let header_size = u16::from_le_bytes([image[6], image[7]]) as usize;
    let fat_size = u64::from_le_bytes(image[8..16].try_into().expect("8 bytes")) as usize;
    let end = header_size
        .checked_add(fat_size)
        .filter(|end| *end <= image.len())
        .ok_or_else(|| bad("size exceeds the image"))?;
    let mut entries = Vec::new();
    let mut at = header_size;
    while at < end {
        let head = image
            .get(at..at + 64)
            .ok_or_else(|| bad("truncated entry header"))?;
        let kind = u16::from_le_bytes([head[0], head[1]]);
        let entry_header = u32::from_le_bytes(head[4..8].try_into().expect("4 bytes")) as usize;
        let size = u64::from_le_bytes(head[8..16].try_into().expect("8 bytes")) as usize;
        let compressed_size = u32::from_le_bytes(head[16..20].try_into().expect("4 bytes"));
        let arch = u32::from_le_bytes(head[28..32].try_into().expect("4 bytes"));
        let decompressed_size = u64::from_le_bytes(head[56..64].try_into().expect("8 bytes"));
        if entry_header < 64 {
            return Err(bad("entry header smaller than 64 bytes"));
        }
        let start = at
            .checked_add(entry_header)
            .ok_or_else(|| bad("entry offset overflow"))?;
        let stop = start
            .checked_add(size)
            .filter(|stop| *stop <= end)
            .ok_or_else(|| bad("entry payload exceeds the image"))?;
        entries.push(FatbinEntry {
            kind,
            arch,
            compressed: compressed_size != 0 || decompressed_size != 0,
            payload: &image[start..stop],
        });
        at = stop;
    }
    Ok(entries)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn fake_cubin(sm: u32) -> Vec<u8> {
        let mut elf = vec![0u8; 64];
        elf[..4].copy_from_slice(ELF_MAGIC);
        elf[4] = 2;
        elf[5] = 1;
        elf[48..52].copy_from_slice(&(sm << 8).to_le_bytes());
        elf
    }

    fn fatbin(entries: &[(u16, u32, &[u8], bool)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (kind, arch, payload, compressed) in entries {
            let mut head = vec![0u8; 64];
            head[0..2].copy_from_slice(&kind.to_le_bytes());
            head[4..8].copy_from_slice(&64u32.to_le_bytes());
            head[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
            if *compressed {
                head[16..20].copy_from_slice(&(payload.len() as u32).to_le_bytes());
            }
            head[28..32].copy_from_slice(&arch.to_le_bytes());
            body.extend_from_slice(&head);
            body.extend_from_slice(payload);
        }
        let mut image = Vec::new();
        image.extend_from_slice(&FATBIN_MAGIC.to_le_bytes());
        image.extend_from_slice(&1u16.to_le_bytes());
        image.extend_from_slice(&16u16.to_le_bytes());
        image.extend_from_slice(&(body.len() as u64).to_le_bytes());
        image.extend_from_slice(&body);
        image
    }

    fn no_ptxas() -> PtxasLocator {
        PtxasLocator::Explicit(PathBuf::from("/nonexistent/ptxas"))
    }

    #[test]
    fn classifies_the_three_image_kinds() {
        assert_eq!(classify(&fake_cubin(120)).unwrap(), ImageKind::Cubin);
        assert_eq!(
            classify(b".version 6.3\n.target sm_50\n.address_size 64\n\0").unwrap(),
            ImageKind::Ptx
        );
        assert_eq!(classify(&fatbin(&[])).unwrap(), ImageKind::Fatbin);
        assert_eq!(classify(b"hello").unwrap_err().code, cu::INVALID_IMAGE);
    }

    #[test]
    fn matching_cubin_passes_through_and_old_cubin_is_refused() {
        let ok = prepare(&fake_cubin(120), JitOptions::default(), &no_ptxas()).unwrap();
        assert!(!ok.compiled);
        let old = prepare(&fake_cubin(61), JitOptions::default(), &no_ptxas()).unwrap_err();
        assert_eq!(old.code, cu::NO_BINARY_FOR_GPU);
        assert!(old.reason.contains("sm_61"));
    }

    #[test]
    fn fatbin_prefers_matching_cubin_and_reports_what_it_holds() {
        let good = fake_cubin(120);
        let old = fake_cubin(52);
        let image = fatbin(&[
            (FATBIN_KIND_ELF, 52, &old, false),
            (FATBIN_KIND_ELF, 120, &good, false),
        ]);
        assert_eq!(
            prepare(&image, JitOptions::default(), &no_ptxas())
                .unwrap()
                .cubin,
            good
        );

        let only_old = fatbin(&[
            (FATBIN_KIND_ELF, 52, &old, false),
            (FATBIN_KIND_PTX, 52, b"zz", true),
        ]);
        let err = prepare(&only_old, JitOptions::default(), &no_ptxas()).unwrap_err();
        assert_eq!(err.code, cu::NO_BINARY_FOR_GPU);
        assert!(err.reason.contains("elf:sm_52") && err.reason.contains("(compressed)"));
    }

    #[test]
    fn truncated_fatbin_is_an_invalid_image() {
        let mut image = fatbin(&[(FATBIN_KIND_ELF, 120, &fake_cubin(120), false)]);
        image.truncate(image.len() - 8);
        assert_eq!(fatbin_entries(&image).unwrap_err().code, cu::INVALID_IMAGE);
    }

    #[test]
    fn ptx_goes_through_the_located_compiler() {
        use std::os::unix::fs::PermissionsExt;
        let dir = private_temp_dir().unwrap();
        let fake = dir.join("fake-ptxas");
        let cubin = dir.join("canned.cubin");
        std::fs::write(&cubin, fake_cubin(120)).unwrap();
        // Fake compiler: checks the flags it was given and copies a canned cubin.
        std::fs::write(
            &fake,
            format!(
                "#!/bin/sh\n[ \"$1\" = --gpu-name=sm_120 ] || exit 9\n[ \"$2\" = -O2 ] || exit 8\necho note >&2\ncp '{}' \"$5\"\n",
                cubin.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o700)).unwrap();
        let ptx = b".version 6.3\n.target sm_50\n.address_size 64\n\0";
        let out = prepare(
            ptx,
            JitOptions { opt_level: 2 },
            &PtxasLocator::Explicit(fake),
        )
        .unwrap();
        assert!(out.compiled);
        assert_eq!(out.cubin, fake_cubin(120));
        assert!(out.log.contains("note"));
        let missing = prepare(ptx, JitOptions::default(), &no_ptxas()).unwrap_err();
        assert_eq!(missing.code, cu::NO_BINARY_FOR_GPU);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
