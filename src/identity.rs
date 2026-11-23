// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: compiler identity + per-cubin manifest for
// ptxas-darwin-arm64 (portkit solution 105, Route B). Host-independent so the
// JSON shapes are unit-tested on any platform.
//
// Identity JSON (`ptxas-darwin-arm64 --identity`, schema cuda-oxide.ptxas-identity.v1):
//   wrapper, wrapper_version, redist_version, redist_platform,
//   archive_path, archive_sha256, archive_sha256_pinned, archive_pinned,
//   x18_manifest_path, x18_manifest_sha256, x18_manifest_sha256_pinned, x18_manifest_pinned,
//   loader_git_describe (string|null),
//   binary_path, binary_sha256, verified, problems[]
// Manifest JSON (`<out>.json` next to the cubin, schema cuda-oxide.ptxas-manifest.v1):
//   input, output, ptx_sha256, ptx_bytes, cubin_sha256, cubin_bytes, arch, cubin_sm,
//   options[], kernels[], identity{...}

use std::path::{Path, PathBuf};

use crate::{default_archive, default_manifest, REDIST_VERSION};

/// SHA-256 of the one archive the x18 manifest is generated for (linux-sbsa 13.2.51).
pub const ARCHIVE_SHA256: &str = "d703609da635db3976a3bc228f34e62f66acffcdc6d4bc5df09ba3a383e716cd";
/// SHA-256 of the x18 rewrite manifest `scripts/gen_x18_patches.py` generates from that
/// archive. The generator is deterministic (re-generated 2026-11-09: identical bytes), so a
/// different value means a different generator or archive.
pub const X18_MANIFEST_SHA256: &str =
    "5b0104356d6c62e03ae64b64ae75a9f75265b151f2da9da45df097085fd060e3";
pub const IDENTITY_SCHEMA: &str = "cuda-oxide.ptxas-identity.v1";
pub const MANIFEST_SCHEMA: &str = "cuda-oxide.ptxas-manifest.v1";
pub const WRAPPER_NAME: &str = "ptxas-darwin-arm64";
pub const WRAPPER_VERSION: &str = env!("CARGO_PKG_VERSION");
/// `git describe --always --dirty` of the repo at build time (build.rs), if git was available.
pub const LOADER_GIT_DESCRIBE: Option<&str> = option_env!("PTXAS_NATIVE_GIT_DESCRIBE");
/// Env var that makes every compile also write `<out>.json`.
pub const MANIFEST_ENV: &str = "CUDA_OXIDE_PTXAS_MANIFEST";

// ---------------------------------------------------------------------------
// SHA-256: CommonCrypto on macOS (hardware-accelerated; the archive is ~96 MB),
// a portable FIPS 180-4 implementation elsewhere and as the test oracle.
// ---------------------------------------------------------------------------

/// Portable SHA-256 (FIPS 180-4).
pub fn sha256_portable(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[cfg(target_os = "macos")]
fn sha256_raw(data: &[u8]) -> [u8; 32] {
    #[repr(C)]
    struct CcSha256Ctx {
        _opaque: [u32; 26], // CC_SHA256_CTX: count[2] + hash[8] + wbuf[16]
    }
    extern "C" {
        fn CC_SHA256_Init(c: *mut CcSha256Ctx) -> i32;
        fn CC_SHA256_Update(c: *mut CcSha256Ctx, data: *const u8, len: u32) -> i32;
        fn CC_SHA256_Final(md: *mut u8, c: *mut CcSha256Ctx) -> i32;
    }
    let mut ctx = CcSha256Ctx { _opaque: [0; 26] };
    let mut md = [0u8; 32];
    // SAFETY: CommonCrypto (libSystem) with a correctly sized context; chunks < 4 GiB.
    unsafe {
        CC_SHA256_Init(&mut ctx);
        for chunk in data.chunks(1 << 30) {
            CC_SHA256_Update(&mut ctx, chunk.as_ptr(), chunk.len() as u32);
        }
        CC_SHA256_Final(md.as_mut_ptr(), &mut ctx);
    }
    md
}

#[cfg(not(target_os = "macos"))]
fn sha256_raw(data: &[u8]) -> [u8; 32] {
    sha256_portable(data)
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Lower-case hex SHA-256.
pub fn sha256_hex(data: &[u8]) -> String {
    hex(&sha256_raw(data))
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    Ok(sha256_hex(&std::fs::read(path)?))
}

// ---------------------------------------------------------------------------
// JSON (no serde: this crate has no registry dependencies)
// ---------------------------------------------------------------------------

pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_opt(s: Option<&str>) -> String {
    s.map(json_str).unwrap_or_else(|| "null".into())
}

fn json_list(v: &[String]) -> String {
    format!(
        "[{}]",
        v.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(", ")
    )
}

fn path_str(p: &Path) -> String {
    std::fs::canonicalize(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub archive_path: PathBuf,
    pub archive_sha256: Option<String>,
    pub x18_manifest_path: PathBuf,
    pub x18_manifest_sha256: Option<String>,
    pub binary_path: Option<PathBuf>,
    pub binary_sha256: Option<String>,
    pub loader_git_describe: Option<String>,
    pub problems: Vec<String>,
}

impl Identity {
    /// Identity from already-known hashes (the loader computes them from the bytes it links).
    pub fn from_parts(
        archive_path: &Path,
        archive_sha256: Option<String>,
        x18_manifest_path: &Path,
        x18_manifest_sha256: Option<String>,
    ) -> Self {
        let mut problems = Vec::new();
        match &archive_sha256 {
            None => problems.push(format!("archive unreadable: {}", archive_path.display())),
            Some(s) if s != ARCHIVE_SHA256 => problems.push(format!(
                "archive sha256 {s} != pinned {ARCHIVE_SHA256} (the x18 manifest is archive-specific)"
            )),
            _ => {}
        }
        match &x18_manifest_sha256 {
            None => problems.push(format!(
                "x18 manifest unreadable: {} (run portkit/ptxas-native/build.sh)",
                x18_manifest_path.display()
            )),
            Some(s) if s != X18_MANIFEST_SHA256 => problems.push(format!(
                "x18 manifest sha256 {s} != pinned {X18_MANIFEST_SHA256} (regenerate with build.sh)"
            )),
            _ => {}
        }
        let binary_path = std::env::current_exe().ok();
        let binary_sha256 = binary_path.as_deref().and_then(|p| sha256_file(p).ok());
        Identity {
            archive_path: archive_path.to_path_buf(),
            archive_sha256,
            x18_manifest_path: x18_manifest_path.to_path_buf(),
            x18_manifest_sha256,
            binary_path,
            binary_sha256,
            loader_git_describe: LOADER_GIT_DESCRIBE.map(str::to_string),
            problems,
        }
    }

    /// Identity of the default (env-overridable) archive + manifest, hashing both files.
    pub fn current() -> Self {
        let a = default_archive();
        let m = default_manifest();
        let ash = sha256_file(&a).ok();
        let msh = sha256_file(&m).ok();
        Self::from_parts(&a, ash, &m, msh)
    }

    pub fn archive_pinned(&self) -> bool {
        self.archive_sha256.as_deref() == Some(ARCHIVE_SHA256)
    }

    pub fn x18_manifest_pinned(&self) -> bool {
        self.x18_manifest_sha256.as_deref() == Some(X18_MANIFEST_SHA256)
    }

    /// True when both the archive and the x18 manifest are the pinned ones.
    pub fn verified(&self) -> bool {
        self.problems.is_empty()
    }

    pub fn to_json(&self, indent: &str) -> String {
        let i = indent;
        let fields = [
            ("schema", json_str(IDENTITY_SCHEMA)),
            ("wrapper", json_str(WRAPPER_NAME)),
            ("wrapper_version", json_str(WRAPPER_VERSION)),
            ("redist_version", json_str(REDIST_VERSION)),
            ("redist_platform", json_str("linux-sbsa")),
            ("archive_path", json_str(&path_str(&self.archive_path))),
            ("archive_sha256", json_opt(self.archive_sha256.as_deref())),
            ("archive_sha256_pinned", json_str(ARCHIVE_SHA256)),
            ("archive_pinned", self.archive_pinned().to_string()),
            (
                "x18_manifest_path",
                json_str(&path_str(&self.x18_manifest_path)),
            ),
            (
                "x18_manifest_sha256",
                json_opt(self.x18_manifest_sha256.as_deref()),
            ),
            ("x18_manifest_sha256_pinned", json_str(X18_MANIFEST_SHA256)),
            (
                "x18_manifest_pinned",
                self.x18_manifest_pinned().to_string(),
            ),
            (
                "loader_git_describe",
                json_opt(self.loader_git_describe.as_deref()),
            ),
            (
                "binary_path",
                json_opt(self.binary_path.as_deref().map(path_str).as_deref()),
            ),
            ("binary_sha256", json_opt(self.binary_sha256.as_deref())),
            ("verified", self.verified().to_string()),
            ("problems", json_list(&self.problems)),
        ];
        let body = fields
            .iter()
            .map(|(k, v)| format!("{i}  {}: {v}", json_str(k)))
            .collect::<Vec<_>>()
            .join(",\n");
        format!("{{\n{body}\n{i}}}")
    }
}

// ---------------------------------------------------------------------------
// Per-cubin manifest
// ---------------------------------------------------------------------------

/// `--gpu-name=sm_XX` value from normalized options.
pub fn arch_of(options: &[String]) -> Option<String> {
    options
        .iter()
        .rev()
        .find_map(|o| o.strip_prefix("--gpu-name=").map(str::to_string))
}

/// Manifest path for an output cubin: `<out>.json` (e.g. `k.cubin` -> `k.cubin.json`).
pub fn manifest_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".json");
    PathBuf::from(s)
}

/// Absolute form of a path that may not exist yet (the cubin before it is published).
pub fn abs_path(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let parent = p
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    match (std::fs::canonicalize(parent), p.file_name()) {
        (Ok(d), Some(n)) => d.join(n),
        _ => std::env::current_dir()
            .map(|d| d.join(p))
            .unwrap_or_else(|_| p.to_path_buf()),
    }
}

/// Manifest for a cubin that already passed [`crate::validate::validate_cubin`].
pub fn manifest_json(
    input: &Path,
    output: &Path,
    ptx: &[u8],
    cubin: &[u8],
    options: &[String],
    validated: &crate::validate::Validated,
    id: &Identity,
) -> String {
    format!(
        "{{\n  \"schema\": {},\n  \"input\": {},\n  \"output\": {},\n  \"ptx_sha256\": {},\n  \"ptx_bytes\": {},\n  \"cubin_sha256\": {},\n  \"cubin_bytes\": {},\n  \"arch\": {},\n  \"cubin_sm\": {},\n  \"options\": {},\n  \"kernels\": {},\n  \"identity\": {}\n}}\n",
        json_str(MANIFEST_SCHEMA),
        json_str(&abs_path(input).to_string_lossy()),
        json_str(&abs_path(output).to_string_lossy()),
        json_str(&sha256_hex(ptx)),
        ptx.len(),
        json_str(&sha256_hex(cubin)),
        cubin.len(),
        json_opt(arch_of(options).as_deref()),
        validated.sm,
        json_list(options),
        json_list(&validated.kernels),
        id.to_json("  "),
    )
}

fn temp_sibling(path: &Path) -> PathBuf {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".into());
    dir.join(format!(".{name}.tmp.{}", std::process::id()))
}

fn remove_if_exists(p: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(p) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Write via a sibling temp file + rename, so readers never see a partial file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_sibling(path);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Publish a validated cubin and, optionally, its manifest as a PAIR:
///  1. any existing `<out>.json` is removed first, so a manifest never describes a
///     different cubin than the one next to it;
///  2. both files are fully written to temp siblings before anything is renamed;
///  3. the cubin is renamed into place, then the manifest; if the manifest rename
///     fails the cubin is removed again.
///
/// A reader therefore sees either no manifest or a manifest whose `cubin_sha256`
/// matches the cubin. Without a manifest request, a stale `<out>.json` is removed.
pub fn publish(out: &Path, cubin: &[u8], manifest: Option<&str>) -> std::io::Result<()> {
    let mpath = manifest_path(out);
    remove_if_exists(&mpath)?;
    let ctmp = temp_sibling(out);
    let mtmp = temp_sibling(&mpath);
    let staged = std::fs::write(&ctmp, cubin).and_then(|_| match manifest {
        Some(m) => std::fs::write(&mtmp, m),
        None => Ok(()),
    });
    let cleanup = || {
        let _ = std::fs::remove_file(&ctmp);
        let _ = std::fs::remove_file(&mtmp);
    };
    if let Err(e) = staged {
        cleanup();
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&ctmp, out) {
        cleanup();
        return Err(e);
    }
    if manifest.is_some() {
        if let Err(e) = std::fs::rename(&mtmp, &mpath) {
            cleanup();
            let _ = std::fs::remove_file(out);
            return Err(e);
        }
    }
    Ok(())
}

/// Remove `<out>` and `<out>.json` after a failed compile/validation so no stale
/// output from an earlier run can be mistaken for this run's result.
pub fn unpublish(out: &Path) {
    let _ = std::fs::remove_file(out);
    let _ = std::fs::remove_file(manifest_path(out));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        let e = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(hex(&sha256_portable(b"")), e);
        assert_eq!(sha256_hex(b""), e);
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(hex(&sha256_portable(b"abc")), abc);
        assert_eq!(sha256_hex(b"abc"), abc);
        let two = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let two_h = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
        assert_eq!(hex(&sha256_portable(two)), two_h);
        // platform backend agrees with the portable one across block boundaries
        let big: Vec<u8> = (0..200_003u32).map(|i| (i * 31 % 251) as u8).collect();
        assert_eq!(sha256_hex(&big), hex(&sha256_portable(&big)));
    }

    #[test]
    fn json_escaping() {
        assert_eq!(json_str("a\"b\\c\n\u{1}"), "\"a\\\"b\\\\c\\n\\u0001\"");
        assert_eq!(json_list(&["x".into(), "y".into()]), "[\"x\", \"y\"]");
    }

    #[test]
    fn manifest_path_and_arch() {
        assert_eq!(
            manifest_path(Path::new("/t/k.cubin")),
            PathBuf::from("/t/k.cubin.json")
        );
        let o = vec!["--gpu-name=sm_120".to_string(), "-O3".to_string()];
        assert_eq!(arch_of(&o).as_deref(), Some("sm_120"));
        assert_eq!(arch_of(&["-O3".to_string()]), None);
    }

    #[test]
    fn identity_flags_unpinned_archive() {
        let id = Identity::from_parts(
            Path::new("/nonexistent/a"),
            Some("00".repeat(32)),
            Path::new("/nonexistent/m"),
            Some("11".repeat(32)),
        );
        assert!(!id.archive_pinned());
        assert!(!id.verified());
        let j = id.to_json("");
        assert!(j.contains("\"verified\": false"), "{j}");
        assert!(j.contains("\"archive_pinned\": false"));
        assert!(j.contains(ARCHIVE_SHA256));
        let ok = Identity::from_parts(
            Path::new("/a"),
            Some(ARCHIVE_SHA256.into()),
            Path::new("/m"),
            Some(X18_MANIFEST_SHA256.into()),
        );
        assert!(ok.verified(), "{:?}", ok.problems);
        let missing = Identity::from_parts(Path::new("/a"), None, Path::new("/m"), None);
        assert_eq!(missing.problems.len(), 2);
        let wrong_x18 = Identity::from_parts(
            Path::new("/a"),
            Some(ARCHIVE_SHA256.into()),
            Path::new("/m"),
            Some("22".repeat(32)),
        );
        assert!(!wrong_x18.verified() && wrong_x18.archive_pinned());
    }

    #[test]
    fn manifest_has_hashes() {
        let id = Identity::from_parts(
            Path::new("/a"),
            Some(ARCHIVE_SHA256.into()),
            Path::new("/m"),
            Some("11".repeat(32)),
        );
        let v = crate::validate::Validated {
            sm: 120,
            kernels: vec!["k".into()],
        };
        let m = manifest_json(
            Path::new("k.ptx"),
            Path::new("k.cubin"),
            b"abc",
            b"not-elf",
            &["--gpu-name=sm_120".into()],
            &v,
            &id,
        );
        assert!(m.contains(
            "\"ptx_sha256\": \"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\""
        ));
        assert!(m.contains("\"arch\": \"sm_120\""));
        assert!(m.contains("\"cubin_sm\": 120"));
        assert!(m.contains("\"kernels\": [\"k\"]"));
        assert!(m.contains(MANIFEST_SCHEMA) && m.contains(IDENTITY_SCHEMA));
    }

    #[test]
    fn publish_pair_and_stale_manifest() {
        let d = std::env::temp_dir().join(format!("ptxas-native-pub-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let out = d.join("k.cubin");
        publish(&out, b"c1", Some("{\"m\":1}")).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"c1");
        assert_eq!(std::fs::read(manifest_path(&out)).unwrap(), b"{\"m\":1}");
        // republish without a manifest: the stale manifest must go
        publish(&out, b"c2", None).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"c2");
        assert!(!manifest_path(&out).exists());
        // a failed publish (target dir missing) leaves nothing half-written
        let bad = d.join("missing/k.cubin");
        assert!(publish(&bad, b"x", Some("{}")).is_err());
        unpublish(&out);
        assert!(!out.exists());
        assert_eq!(
            std::fs::read_dir(&d).unwrap().count(),
            0,
            "no temp files left"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn atomic_write_replaces() {
        let d = std::env::temp_dir().join(format!("ptxas-native-aw-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("x.cubin");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"two");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
