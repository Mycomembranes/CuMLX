// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: parser for the offline patch manifest written by
// scripts/gen_patch_manifest.py. Parsing is strict and fail-closed: an unknown
// row kind, a `svc` row, a duplicate address or a foreign ELF hash is an error.

use crate::a64::X18Shape;

pub const SCHEMA: &str = "cuda-oxide.tileiras-native-patches.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchKind {
    X18(X18Shape),
    Tls,
    Abs64,
    /// Aligned in-range data word the rebase rule must leave unslid (its value
    /// is contradicted as an address). old_word / new_word = low / high half.
    NoRebase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchRow {
    /// Link-time virtual address.
    pub address: u64,
    pub kind: PatchKind,
    pub old_word: u32,
    /// x18: instruction with x18 -> R. abs64: high half of the 64-bit value.
    pub new_word: u32,
    pub r: u32,
    pub r2: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebaseRange {
    pub section: String,
    pub address: u64,
    pub size: u64,
    pub count: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub elf_sha256: String,
    pub image_lo: u64,
    pub image_hi: u64,
    pub rebase: Vec<RebaseRange>,
    pub rows: Vec<PatchRow>,
}

impl Manifest {
    pub fn count(&self, pred: impl Fn(&PatchKind) -> bool) -> usize {
        self.rows.iter().filter(|r| pred(&r.kind)).count()
    }
}

fn hex(text: &str) -> Result<u64, String> {
    let digits = text
        .strip_prefix("0x")
        .ok_or_else(|| format!("expected 0x-prefixed hex: {text}"))?;
    u64::from_str_radix(digits, 16).map_err(|_| format!("bad hex {text}"))
}
fn reg(text: &str) -> Result<u32, String> {
    let n: u32 = text
        .strip_prefix('x')
        .and_then(|d| d.parse().ok())
        .ok_or_else(|| format!("bad register {text}"))?;
    if n > 30 {
        return Err(format!("bad register {text}"));
    }
    Ok(n)
}

pub fn parse(text: &str) -> Result<Manifest, String> {
    let mut lines = text.lines();
    if lines.next() != Some(&format!("# {SCHEMA}")[..]) {
        return Err("patch manifest schema line missing".into());
    }
    let mut manifest = Manifest {
        elf_sha256: String::new(),
        image_lo: 0,
        image_hi: 0,
        rebase: Vec::new(),
        rows: Vec::new(),
    };
    let mut declared: Option<Vec<(String, u64)>> = None;
    for (index, line) in lines.enumerate() {
        let at = |m: &str| format!("patch manifest line {}: {m}", index + 2);
        if let Some(header) = line.strip_prefix("# ") {
            if let Some(v) = header.strip_prefix("elf_sha256=") {
                manifest.elf_sha256 = v.to_string();
            } else if let Some(v) = header.strip_prefix("image_lo=") {
                let (lo, hi) = v
                    .split_once(" image_hi=")
                    .ok_or_else(|| at("bad image bounds"))?;
                manifest.image_lo = hex(lo).map_err(|e| at(&e))?;
                manifest.image_hi = hex(hi).map_err(|e| at(&e))?;
            } else if let Some(v) = header.strip_prefix("counts ") {
                let mut counts = Vec::new();
                for pair in v.split(' ') {
                    let (k, n) = pair.split_once('=').ok_or_else(|| at("bad count"))?;
                    counts.push((k.to_string(), n.parse().map_err(|_| at("bad count"))?));
                }
                declared = Some(counts);
            } else if let Some(v) = header.strip_prefix("rebase ") {
                let f: Vec<&str> = v.split(' ').collect();
                if f.len() != 5 || f[4].len() != 64 {
                    return Err(at("bad rebase header"));
                }
                manifest.rebase.push(RebaseRange {
                    section: f[0].to_string(),
                    address: hex(f[1]).map_err(|e| at(&e))?,
                    size: hex(f[2]).map_err(|e| at(&e))?,
                    count: f[3].parse().map_err(|_| at("bad rebase count"))?,
                    sha256: f[4].to_string(),
                });
            }
            continue;
        }
        if line.is_empty() {
            return Err(at("empty line"));
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 6 {
            return Err(at("expected 6 fields"));
        }
        let kind = match f[1] {
            "x18:plain" => PatchKind::X18(X18Shape::Plain),
            "x18:adrp" => PatchKind::X18(X18Shape::Adrp),
            "x18:cb" => PatchKind::X18(X18Shape::Cb),
            "x18:tb" => PatchKind::X18(X18Shape::Tb),
            "x18:blr" => PatchKind::X18(X18Shape::Blr),
            "tls" => PatchKind::Tls,
            "abs64" => PatchKind::Abs64,
            "norebase" => PatchKind::NoRebase,
            "svc" => return Err(at("direct system call site: no port exists")),
            other => return Err(at(&format!("unknown patch kind {other}"))),
        };
        let word = |text: &str| -> Result<u32, String> {
            u32::try_from(hex(text).map_err(|e| at(&e))?).map_err(|_| at("word wider than 32 bits"))
        };
        let row = PatchRow {
            address: hex(f[0]).map_err(|e| at(&e))?,
            kind,
            old_word: word(f[2])?,
            new_word: word(f[3])?,
            r: reg(f[4]).map_err(|e| at(&e))?,
            r2: reg(f[5]).map_err(|e| at(&e))?,
        };
        if row.address % 4 != 0 {
            return Err(at("unaligned address"));
        }
        if let Some(last) = manifest.rows.last() {
            let width = if matches!(last.kind, PatchKind::Abs64 | PatchKind::NoRebase) {
                8
            } else {
                4
            };
            if row.address < last.address + width {
                return Err(at("rows overlap or are not sorted"));
            }
        }
        match row.kind {
            PatchKind::X18(_) => {
                if !(9..=15).contains(&row.r) || !(9..=15).contains(&row.r2) || row.r == row.r2 {
                    return Err(at("illegal scratch register"));
                }
                // new_word must be old_word with some register fields 18 -> R
                // and nothing else changed; R and R2 must not already be
                // operands of the original instruction.
                let mut changed = 0;
                for field in [0u32, 5, 10, 16] {
                    let (o, n) = ((row.old_word >> field) & 31, (row.new_word >> field) & 31);
                    if o == 18 && n == row.r {
                        changed |= 31 << field;
                    } else if o != n {
                        return Err(at("rewrite changes more than x18"));
                    }
                }
                if changed == 0 || (row.old_word ^ row.new_word) & !changed != 0 {
                    return Err(at("rewrite is not an x18 register substitution"));
                }
            }
            PatchKind::Tls => {
                if row.old_word != crate::a64::mrs_tpidr(row.r) || row.r == 18 {
                    return Err(at("tls row is not `mrs xN, tpidr_el0`"));
                }
            }
            PatchKind::Abs64 => {}
            PatchKind::NoRebase => {
                let value = (row.new_word as u64) << 32 | row.old_word as u64;
                if row.address % 8 != 0 || !(manifest.image_lo..manifest.image_hi).contains(&value)
                {
                    return Err(at("norebase row is not an aligned in-image value"));
                }
            }
        }
        manifest.rows.push(row);
    }
    if manifest.elf_sha256.len() != 64 || manifest.image_hi <= manifest.image_lo {
        return Err("patch manifest header incomplete".into());
    }
    let declared = declared.ok_or("patch manifest counts header missing")?;
    for (name, want) in declared {
        let have = match name.as_str() {
            "x18" => manifest.count(|k| matches!(k, PatchKind::X18(_))),
            "tls" => manifest.count(|k| *k == PatchKind::Tls),
            "abs64" => manifest.count(|k| *k == PatchKind::Abs64),
            "norebase" => manifest.count(|k| *k == PatchKind::NoRebase),
            "svc" => 0,
            // informational census values with no rows of their own
            "other_mrs" | "x18_outside_fde" => continue,
            other => return Err(format!("patch manifest declares unknown count {other}")),
        } as u64;
        if have != want {
            return Err(format!(
                "patch manifest declares {want} {name} rows, has {have}"
            ));
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "# cuda-oxide.tileiras-native-patches.v1\n# elf_sha256=297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a\n# image_lo=0x400000 image_hi=0x5b1b3b0\n";

    fn doc(counts: &str, rows: &str) -> String {
        format!(
            "{HEAD}# counts {counts}\n# rebase .got 0x5917828 0x2a7c0 3 {}\n# addr\tkind\n{rows}",
            "a".repeat(64)
        )
    }

    #[test]
    fn accepts_every_row_kind() {
        let m = parse(&doc(
            "abs64=1 svc=0 tls=1 x18=2",
            "0x407624\tx18:plain\t0x910b23f2\t0x910b23e9\tx9\tx10\n0x407628\tx18:blr\t0xd63f0240\t0xd63f0120\tx9\tx10\n0x40762c\ttls\t0xd53bd048\t0x00000000\tx8\tx0\n0x4fec40\tabs64\t0x00407060\t0x00000000\tx0\tx0\n",
        ))
        .unwrap();
        assert_eq!(m.rows.len(), 4);
        assert_eq!(m.rebase[0].count, 3);
        assert_eq!(m.rows[0].kind, PatchKind::X18(X18Shape::Plain));
    }

    #[test]
    fn refuses_svc_unknown_unsorted_and_miscounted() {
        let row = "0x407624\tx18:plain\t0x910b23f2\t0x910b23e9\tx9\tx10\n";
        assert!(parse(&doc(
            "x18=1",
            "0x407624\tsvc\t0xd4000001\t0x00000000\tx0\tx0\n"
        ))
        .is_err());
        assert!(parse(&doc("x18=1", &row.replace("plain", "novel"))).is_err());
        assert!(parse(&doc("x18=2", &format!("{row}{row}"))).is_err());
        assert!(parse(&doc("x18=2", row)).is_err());
        assert!(parse(&doc("x18=1", &row.replace("x10", "x9"))).is_err());
        assert!(parse(&doc("x18=1", &row.replace("x9", "x16"))).is_err());
        assert!(parse(&doc(
            "tls=1",
            "0x40762c\ttls\t0xd53bd048\t0x00000000\tx9\tx0\n"
        ))
        .is_err());
        assert!(parse(&row.to_string()).is_err());
        assert!(parse(&doc("x18=1 novel=3", row)).is_err());
        assert!(parse(&doc("x18=1", &row.replace("0x910b23e9", "0x910b23ea"))).is_err());
        assert!(parse(&doc("x18=1", &row.replace("0x910b23e9", "0x910b27e9"))).is_err());
        assert!(parse(&doc("x18=1", &row.replace("0x910b23f2", "0x1910b23f2"))).is_err());
        assert!(parse(&doc("x18=1 other_mrs=12", row)).is_ok());
        // norebase: aligned address, value inside the image, counted.
        let keep = "0x5810360\tnorebase\t0x01000000\t0x00000000\tx0\tx0\n";
        assert_eq!(
            parse(&doc("norebase=1", keep)).unwrap().rows[0].kind,
            PatchKind::NoRebase
        );
        assert!(parse(&doc("norebase=1", &keep.replace("0x5810360", "0x5810364"))).is_err());
        assert!(parse(&doc(
            "norebase=1",
            &keep.replace("0x01000000", "0x00000010")
        ))
        .is_err());
        assert!(parse(&doc(
            "norebase=1",
            &keep.replace("0x00000000\tx0", "0x00000001\tx0")
        ))
        .is_err());
        assert!(parse(&doc("norebase=0", keep)).is_err());
    }
}
