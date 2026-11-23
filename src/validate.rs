// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: validate a cubin produced by ptxas-darwin-arm64
// BEFORE it is published (portkit solution 105, Route B). A cubin is written
// only if it is an ET_EXEC EM_CUDA ELF for the requested SM, every PTX `.entry`
// is present with a non-empty `.text`, its `.nv.info.<k>` parses strictly, and
// its KPARAM_INFO records are contiguous, non-overlapping, inside
// CBANK_PARAM_SIZE and match the PTX parameter sizes one for one.
// Host-independent; mirrors scripts/cubin_diff.py `check --ptx`.

use std::collections::BTreeMap;

use crate::{err, parse_cubin, Error, EM_CUDA};

const EIFMT_NVAL: u8 = 0x01;
const EIFMT_BVAL: u8 = 0x02;
const EIFMT_HVAL: u8 = 0x03;
const EIFMT_SVAL: u8 = 0x04;
const EIATTR_KPARAM_INFO: u8 = 0x17;
const EIATTR_CBANK_PARAM_SIZE: u8 = 0x19;
const ET_EXEC: u16 = 2;

/// One EIATTR record: (format, attribute, 16-bit value, SVAL payload).
pub type NvInfoRecord = (u8, u8, u16, Vec<u8>);

/// Strict `.nv.info` walk: truncated/overrunning records, trailing bytes and unknown
/// formats are errors.
pub fn nv_info_strict(data: &[u8]) -> Result<Vec<NvInfoRecord>, Error> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        if i + 4 > data.len() {
            return Err(err(format!(".nv.info: truncated record header at {i}")));
        }
        let (fmt, attr) = (data[i], data[i + 1]);
        let v = u16::from_le_bytes([data[i + 2], data[i + 3]]);
        match fmt {
            EIFMT_SVAL => {
                let end = i + 4 + v as usize;
                if end > data.len() {
                    return Err(err(format!(
                        ".nv.info: attr {attr:#04x} payload ({v} B) overruns the section at {i}"
                    )));
                }
                out.push((fmt, attr, v, data[i + 4..end].to_vec()));
                i = end;
            }
            EIFMT_NVAL | EIFMT_BVAL | EIFMT_HVAL => {
                out.push((fmt, attr, v, Vec::new()));
                i += 4;
            }
            _ => {
                return Err(err(format!(
                    ".nv.info: unknown record format {fmt:#04x} at {i}"
                )))
            }
        }
    }
    Ok(out)
}

/// KPARAM_INFO of one kernel as (ordinal, offset, size), validated, sorted by ordinal.
pub fn kparams(info: &[u8], kname: &str) -> Result<Vec<(u16, u16, u32)>, Error> {
    let mut params = Vec::new();
    let mut cbank_param_size = None;
    for (fmt, attr, v, payload) in nv_info_strict(info)? {
        if attr == EIATTR_KPARAM_INFO {
            if fmt != EIFMT_SVAL || payload.len() != 12 {
                return Err(err(format!("{kname}: KPARAM_INFO record malformed")));
            }
            let ordinal = u16::from_le_bytes([payload[4], payload[5]]);
            let offset = u16::from_le_bytes([payload[6], payload[7]]);
            let flags = u32::from_le_bytes(payload[8..12].try_into().unwrap());
            params.push((ordinal, offset, (flags >> 18) & 0x3fff));
        } else if attr == EIATTR_CBANK_PARAM_SIZE && fmt == EIFMT_HVAL {
            cbank_param_size = Some(v as u32);
        }
    }
    params.sort();
    if params.iter().enumerate().any(|(i, p)| p.0 as usize != i) {
        return Err(err(format!(
            "{kname}: KPARAM ordinals not contiguous from 0"
        )));
    }
    let mut end = 0u32;
    for &(o, off, size) in &params {
        if size == 0 {
            return Err(err(format!("{kname}: KPARAM {o} has size 0")));
        }
        if (off as u32) < end {
            return Err(err(format!(
                "{kname}: KPARAM {o} at {off} overlaps the previous parameter"
            )));
        }
        end = off as u32 + size;
    }
    if !params.is_empty() {
        match cbank_param_size {
            None => {
                return Err(err(format!(
                    "{kname}: KPARAM_INFO without CBANK_PARAM_SIZE"
                )))
            }
            Some(c) if end > c => {
                return Err(err(format!(
                    "{kname}: parameters end at {end} > CBANK_PARAM_SIZE {c}"
                )))
            }
            _ => {}
        }
    }
    Ok(params)
}

fn strip_comments(ptx: &str) -> String {
    let b = ptx.as_bytes();
    let mut out = String::with_capacity(ptx.len());
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.push(' ');
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn param_size(decl: &str) -> Option<u32> {
    if !decl.starts_with(".param") {
        return None;
    }
    let bits = decl.split_whitespace().find_map(|t| {
        let t = t.strip_prefix('.')?;
        let (k, n) = t.split_at(1);
        (["b", "s", "u", "f"].contains(&k))
            .then(|| n.parse::<u32>().ok())
            .flatten()
            .filter(|n| [8, 16, 32, 64, 128].contains(n))
    })?;
    let count = match decl.trim_end().strip_suffix(']') {
        Some(head) => head.rsplit_once('[')?.1.trim().parse::<u32>().ok()?,
        None => 1,
    };
    Some(bits / 8 * count)
}

/// `{entry name: [param byte sizes]}` from the PTX `.entry` declarations.
pub fn ptx_entry_params(ptx: &[u8]) -> Result<BTreeMap<String, Vec<u32>>, Error> {
    let text = strip_comments(&String::from_utf8_lossy(ptx));
    let mut out = BTreeMap::new();
    let mut rest = text.as_str();
    while let Some(p) = rest.find(".entry") {
        let after = &rest[p + ".entry".len()..];
        if !after.starts_with(char::is_whitespace) {
            rest = after;
            continue;
        }
        let after = after.trim_start();
        let name_len = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || "_$%".contains(c)))
            .unwrap_or(after.len());
        let name = after[..name_len].to_string();
        let tail = after[name_len..].trim_start();
        let Some(body) = tail.strip_prefix('(') else {
            return Err(err(format!("PTX: .entry {name} without a parameter list")));
        };
        let close = body
            .find(')')
            .ok_or_else(|| err(format!("PTX: unterminated parameter list of {name}")))?;
        let mut sizes = Vec::new();
        for decl in body[..close]
            .split(',')
            .map(str::trim)
            .filter(|d| !d.is_empty())
        {
            sizes
                .push(param_size(decl).ok_or_else(|| {
                    err(format!("PTX: cannot size parameter {decl:?} of {name}"))
                })?);
        }
        if out.insert(name.clone(), sizes).is_some() {
            return Err(err(format!("PTX: duplicate entry {name}")));
        }
        rest = &body[close..];
    }
    Ok(out)
}

/// Facts about a cubin that passed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validated {
    pub sm: u32,
    pub kernels: Vec<String>,
}

/// `sm_120` / `sm_120a` / `compute_120` -> 120.
pub fn arch_number(arch: &str) -> Option<u32> {
    let digits: String = arch
        .trim_start_matches("sm_")
        .trim_start_matches("compute_")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Validate `cubin` (compiled from `ptx` for `arch`) before it is published.
pub fn validate_cubin(cubin: &[u8], arch: Option<&str>, ptx: &[u8]) -> Result<Validated, Error> {
    let c = parse_cubin(cubin)?;
    if c.e_type != ET_EXEC {
        return Err(err(format!("e_type {} != ET_EXEC", c.e_type)));
    }
    if c.machine != EM_CUDA {
        return Err(err(format!("e_machine {} != EM_CUDA", c.machine)));
    }
    if let Some(a) = arch {
        let want = arch_number(a).ok_or_else(|| err(format!("unparseable arch {a:?}")))?;
        if c.sm != want {
            return Err(err(format!("cubin is sm_{} but {a} was requested", c.sm)));
        }
    }
    let kernels = c.kernels();
    if kernels.is_empty() {
        return Err(err("no .text.<entry> sections"));
    }
    let entries = ptx_entry_params(ptx)?;
    let mut have = BTreeMap::new();
    for k in &kernels {
        if c.section(&format!(".text.{k}"))
            .is_none_or(|t| t.data.is_empty())
        {
            return Err(err(format!("{k}: empty .text")));
        }
        let info = c
            .section(&format!(".nv.info.{k}"))
            .ok_or_else(|| err(format!("{k}: missing .nv.info.{k}")))?;
        let kp = kparams(&info.data, k)?;
        have.insert(k.clone(), kp.iter().map(|p| p.2).collect::<Vec<_>>());
    }
    if let Some(g) = c.section(".nv.info") {
        nv_info_strict(&g.data)?;
    }
    let want: Vec<&String> = entries.keys().collect();
    let got: Vec<&String> = have.keys().collect();
    if want != got {
        return Err(err(format!(
            "entries {got:?} differ from the PTX .entry set {want:?}"
        )));
    }
    for (k, sizes) in &entries {
        if &have[k] != sizes {
            return Err(err(format!(
                "{k}: KPARAM sizes {:?} != PTX parameter sizes {sizes:?}",
                have[k]
            )));
        }
    }
    Ok(Validated { sm: c.sm, kernels })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kp(ordinal: u16, offset: u16, size: u32) -> Vec<u8> {
        let mut v = vec![0x04, 0x17, 12, 0];
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&ordinal.to_le_bytes());
        v.extend_from_slice(&offset.to_le_bytes());
        v.extend_from_slice(&(size << 18).to_le_bytes());
        v
    }

    fn cbank(size: u16) -> Vec<u8> {
        let mut v = vec![0x03, 0x19];
        v.extend_from_slice(&size.to_le_bytes());
        v
    }

    #[test]
    fn ptx_params_sized() {
        let ptx = b"// c\n.visible .entry k(\n .param .u64 .ptr .align 1 k_param_0, /* x */\n .param .u32 k_param_1,\n .param .align 8 .b8 k_param_2[24]\n)\n{ ret; }\n.entry e2() { ret; }";
        let e = ptx_entry_params(ptx).unwrap();
        assert_eq!(e["k"], vec![8, 4, 24]);
        assert_eq!(e["e2"], Vec::<u32>::new());
        assert!(ptx_entry_params(b".entry k(.param .weird p) {}").is_err());
        assert!(ptx_entry_params(b".entry k(.param .u32 p").is_err());
    }

    #[test]
    fn kparam_rules() {
        let good = [kp(1, 8, 4), kp(0, 0, 8), cbank(12)].concat();
        assert_eq!(kparams(&good, "k").unwrap(), vec![(0, 0, 8), (1, 8, 4)]);
        let gap = [kp(0, 0, 8), kp(2, 8, 4), cbank(12)].concat();
        assert!(kparams(&gap, "k").is_err());
        let overlap = [kp(0, 0, 8), kp(1, 4, 4), cbank(12)].concat();
        assert!(kparams(&overlap, "k").is_err());
        let outside = [kp(0, 0, 8), kp(1, 8, 4), cbank(8)].concat();
        assert!(kparams(&outside, "k").is_err());
        let zero = [kp(0, 0, 0), cbank(8)].concat();
        assert!(kparams(&zero, "k").is_err());
        let no_cbank = kp(0, 0, 8);
        assert!(kparams(&no_cbank, "k").is_err());
    }

    #[test]
    fn nv_info_rejects_truncation() {
        let mut t = kp(0, 0, 8);
        t.truncate(10);
        assert!(nv_info_strict(&t).is_err());
        assert!(nv_info_strict(&[0x03, 0x19]).is_err());
        assert!(nv_info_strict(&[0x07, 0x19, 0, 0]).is_err());
        assert_eq!(nv_info_strict(&cbank(4)).unwrap().len(), 1);
    }

    #[test]
    fn arch_numbers() {
        assert_eq!(arch_number("sm_120"), Some(120));
        assert_eq!(arch_number("sm_120a"), Some(120));
        assert_eq!(arch_number("compute_90"), Some(90));
        assert_eq!(arch_number("bogus"), None);
    }

    #[test]
    fn rejects_non_elf_and_truncated() {
        assert!(validate_cubin(b"nope", Some("sm_120"), b"").is_err());
        let mut h = vec![0u8; 64];
        h[..4].copy_from_slice(b"\x7fELF");
        h[4] = 2;
        h[5] = 1;
        h[40] = 0xff; // e_shoff far out of range
        h[58] = 64;
        h[60] = 3;
        assert!(validate_cubin(&h, None, b"").is_err());
    }
}
