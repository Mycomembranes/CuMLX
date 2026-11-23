// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: offline cross-check of the data-pointer rebase
// rule used to load NVIDIA's relocation-free Linux-aarch64 `tileiras` ET_EXEC
// at `link address + slide`.
//
// The rule (scripts/gen_patch_manifest.py, image.rs) slides every aligned
// 8-byte word of six data sections whose value lies inside the image. An
// ET_EXEC keeps no relocation records, so the rule cannot be proven; this
// module measures how much of it is corroborated by INDEPENDENT evidence and
// how much is not. Nothing is mapped or executed.
//
// First, an in-range word of .data.rel.ro / .data is CONTRADICTED as an
// address, and therefore left unslid (manifest `norebase` rows), when its
// value
//   * lies in no allocated section (the ELF header / padding), or
//   * lies in unwind metadata (.eh_frame_hdr / .eh_frame / .gcc_except_table), or
//   * lies in an executable section without being a function start (FDE
//     initial location, PLT entry, .init, .fini) - this covers every value
//     that is not 4-aligned.
// On the pinned binary these are integers: bit masks 0x400000..0x4000000 and
// the two Itanium-demangler operator tables ({char[2], u8, u8} read as a u64).
// This is evidence from the SHAPE of the value, not proof of how the word is
// used. Unslid, a wrong verdict faults in the unmapped low 4 GiB when the word
// is dereferenced or called, and is reported (a pointer that is only compared
// would not fault); slid, an integer would be changed silently.
//
// Every remaining (slid) word gets the first class that applies:
//
//   pointer_only_section   the word lives in .init_array / .fini_array / .got /
//                          .got.plt, whose every entry is an address by ELF
//                          section typing;
//   function_start         the value is an FDE initial location (.eh_frame_hdr
//                          search table), a PLT entry, or .init / .fini;
//   code_address_target    the value is an address that code itself forms with
//                          ADRP+ADD or ADR (the same object is addressed by
//                          instructions);
//   pointer_bearing_object the value is the 8-aligned address of another
//                          rebased word (an object that starts with a pointer:
//                          vtable groups, type_info, pointer tables);
//   c_string_start         the value points into .rodata right after a NUL (or
//                          at the section start) at a printable, NUL-terminated
//                          string;
//   ambiguous              none of the above: an in-range value with no
//                          corroboration. It may still be a pointer (to .bss,
//                          to a plain .data / .rodata object); it may be an
//                          integer that happens to lie in the image range.
//
// The evidence is conservative, not complete: the ADRP tracker forgets a
// register at every write it recognises and at every call / unconditional
// branch, so it under-reports.

use std::collections::{BTreeMap, HashSet};

/// Sections covered by the rebase rule (mirrors REBASE_SECTIONS in
/// scripts/gen_patch_manifest.py; the manifest binds the resulting word set).
pub const REBASE_SECTIONS: [&str; 6] = [
    ".init_array",
    ".fini_array",
    ".data.rel.ro",
    ".got",
    ".got.plt",
    ".data",
];
const POINTER_ONLY: [&str; 4] = [".init_array", ".fini_array", ".got", ".got.plt"];
const UNWIND: [&str; 3] = [".eh_frame_hdr", ".eh_frame", ".gcc_except_table"];
pub const CONTRADICTIONS: [&str; 4] = [
    "outside_sections",
    "unwind_metadata",
    "executable_unaligned",
    "executable_not_function_start",
];
pub const CLASSES: [&str; 6] = [
    "pointer_only_section",
    "function_start",
    "code_address_target",
    "pointer_bearing_object",
    "c_string_start",
    "ambiguous",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub kind: u32,
    pub flags: u64,
    pub address: u64,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Audit {
    pub image_lo: u64,
    pub image_hi: u64,
    /// Aligned in-range words of the rule's sections (slid + contradicted).
    pub candidates_total: u64,
    /// Words left unslid: (link address, value), sorted. Must equal the
    /// manifest's `norebase` rows.
    pub contradicted: Vec<(u64, u64)>,
    pub contradicted_by_reason: BTreeMap<&'static str, u64>,
    /// Contradicted values that code nevertheless forms with ADRP+ADD / ADR
    /// (evidence against the contradiction; 0 on the pinned binary).
    pub contradicted_but_code_formed: u64,
    /// Words the rule slides.
    pub rebased_total: u64,
    /// Rebased words per containing section.
    pub by_container: BTreeMap<String, u64>,
    /// Rebased words per corroboration class (see the module header).
    pub by_class: BTreeMap<&'static str, u64>,
    /// Rebased words per section their value points into.
    pub by_target_section: BTreeMap<String, u64>,
    pub ambiguous: u64,
    pub ambiguous_by_container: BTreeMap<String, u64>,
    pub ambiguous_by_target_section: BTreeMap<String, u64>,
    /// Slid values inside an executable section that are not a function start
    /// (possible only in the pointer-only sections, which are never contradicted).
    pub executable_target_not_function_start: u64,
    /// Aligned in-range words the rule does NOT rebase, per allocated section
    /// outside the rule.
    pub not_rebased_in_range: BTreeMap<String, u64>,
    /// In-range 8-byte values at unaligned offsets of the rule's sections.
    pub unaligned_in_range_in_rule_sections: u64,
    pub evidence_function_starts: u64,
    pub evidence_code_address_targets: u64,
    pub instructions_scanned: u64,
}

fn rd<const N: usize>(d: &[u8], at: u64) -> Result<[u8; N], String> {
    let at = usize::try_from(at).map_err(|_| "offset overflow")?;
    d.get(at..at.checked_add(N).ok_or("offset overflow")?)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| "read outside the file".to_string())
}
fn u16_at(d: &[u8], at: u64) -> Result<u16, String> {
    Ok(u16::from_le_bytes(rd(d, at)?))
}
fn u32_at(d: &[u8], at: u64) -> Result<u32, String> {
    Ok(u32::from_le_bytes(rd(d, at)?))
}
fn u64_at(d: &[u8], at: u64) -> Result<u64, String> {
    Ok(u64::from_le_bytes(rd(d, at)?))
}

/// Section headers of an ELF64 little-endian file (bounds-checked).
pub fn sections(d: &[u8]) -> Result<Vec<Section>, String> {
    if d.len() < 64 || &d[..6] != b"\x7fELF\x02\x01" {
        return Err("not an ELF64 little-endian file".into());
    }
    let (shoff, shentsize, shnum, shstrndx) = (
        u64_at(d, 40)?,
        u16_at(d, 58)?,
        u16_at(d, 60)? as u64,
        u16_at(d, 62)? as u64,
    );
    if shentsize != 64 || shnum == 0 || shnum > 4096 || shstrndx >= shnum {
        return Err("unsupported section header table".into());
    }
    let raw = |k: u64| -> Result<(u32, u32, u64, u64, u64, u64), String> {
        let p = shoff.checked_add(64 * k).ok_or("section header overflow")?;
        Ok((
            u32_at(d, p)?,
            u32_at(d, p + 4)?,
            u64_at(d, p + 8)?,
            u64_at(d, p + 16)?,
            u64_at(d, p + 24)?,
            u64_at(d, p + 32)?,
        ))
    };
    let (_, _, _, _, stroff, strsize) = raw(shstrndx)?;
    let mut out = Vec::new();
    for k in 0..shnum {
        let (name, kind, flags, address, offset, size) = raw(k)?;
        if name as u64 >= strsize {
            return Err("section name outside the string table".into());
        }
        // The name must start and end inside the section-name string table.
        let table_end = stroff
            .checked_add(strsize)
            .filter(|e| *e <= d.len() as u64)
            .ok_or("section string table outside the file")?;
        let tail = &d[(stroff + name as u64) as usize..table_end as usize];
        let n = tail
            .iter()
            .take(256)
            .position(|b| *b == 0)
            .ok_or("unterminated section name")?;
        if address.checked_add(size).is_none() {
            return Err("section address range overflows".into());
        }
        // SHT_NOBITS occupies no file bytes.
        if kind != 8 && offset.checked_add(size).is_none_or(|e| e > d.len() as u64) {
            return Err("section outside the file".into());
        }
        out.push(Section {
            name: String::from_utf8_lossy(&tail[..n]).into_owned(),
            kind,
            flags,
            address,
            offset,
            size,
        });
    }
    Ok(out)
}

/// Link-time bounds of the PT_LOAD segments.
fn image_bounds(d: &[u8]) -> Result<(u64, u64), String> {
    let (phoff, phnum) = (u64_at(d, 32)?, u16_at(d, 56)? as u64);
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for k in 0..phnum {
        let p = phoff + 56 * k;
        if u32_at(d, p)? == 1 {
            let (va, memsz) = (u64_at(d, p + 16)?, u64_at(d, p + 40)?);
            lo = lo.min(va);
            hi = hi.max(va.checked_add(memsz).ok_or("segment overflow")?);
        }
    }
    if lo >= hi {
        return Err("no PT_LOAD segment".into());
    }
    Ok((lo, hi))
}

/// FDE initial locations from the .eh_frame_hdr binary-search table
/// (version 1, table encoding datarel|sdata4, count encoding udata4).
fn function_starts(d: &[u8], hdr: &Section) -> Result<HashSet<u64>, String> {
    let o = hdr.offset;
    let head: [u8; 4] = rd(d, o)?;
    if head[0] != 1 || head[2] != 0x03 || head[3] != 0x3b {
        return Err(format!(
            ".eh_frame_hdr encoding {head:02x?} is not supported"
        ));
    }
    let ptr_bytes = match head[1] & 0x0f {
        0x03 | 0x0b => 4,
        0x04 | 0x0c => 8,
        _ => return Err(".eh_frame_hdr eh_frame_ptr encoding is not supported".into()),
    };
    let count = u32_at(d, o + 4 + ptr_bytes)? as u64;
    let table = o + 8 + ptr_bytes;
    if table + 8 * count > hdr.offset + hdr.size {
        return Err(".eh_frame_hdr table overruns the section".into());
    }
    let mut out = HashSet::with_capacity(count as usize);
    for k in 0..count {
        let delta = u32_at(d, table + 8 * k)? as i32 as i64;
        out.insert(hdr.address.wrapping_add(delta as u64));
    }
    Ok(out)
}

/// Addresses formed by code with ADRP+ADD or ADR in `words` at `address`.
/// Conservative straight-line tracking: a register's ADRP page is forgotten at
/// every recognised write, call, return and unconditional branch.
pub fn code_address_targets(words: &[u8], address: u64, out: &mut HashSet<u64>) -> u64 {
    let mut page: [Option<u64>; 32] = [None; 32];
    let mut scanned = 0u64;
    for (k, chunk) in words.chunks_exact(4).enumerate() {
        let w = u32::from_le_bytes(chunk.try_into().unwrap());
        let pc = address + 4 * k as u64;
        scanned += 1;
        let (rd, rn, rt2) = (
            (w & 31) as usize,
            ((w >> 5) & 31) as usize,
            ((w >> 10) & 31) as usize,
        );
        if w & 0x1f00_0000 == 0x1000_0000 {
            // ADR / ADRP
            let imm = ((((w >> 5) & 0x7ffff) << 2) | ((w >> 29) & 3)) as i64;
            let imm = (imm << 43) >> 43; // sign-extend 21 bits
            if w >> 31 != 0 {
                page[rd] = Some((pc & !0xfff).wrapping_add((imm << 12) as u64));
            } else {
                out.insert(pc.wrapping_add(imm as u64));
                page[rd] = None;
            }
        } else if w & 0xffc0_0000 == 0x9100_0000 {
            // ADD Xd, Xn, #imm12
            if let Some(p) = page[rn] {
                out.insert(p + ((w >> 10) & 0xfff) as u64);
            }
            page[rd] = None;
        } else if w & 0xfc00_0000 == 0x1400_0000
            || w & 0xfffffc1f == 0xd61f_0000
            || w & 0xfffffc1f == 0xd65f_0000
        {
            page = [None; 32]; // B, BR, RET: the next instruction is not a successor
        } else if w & 0xfc00_0000 == 0x9400_0000 || w & 0xfffffc1f == 0xd63f_0000 {
            for r in page.iter_mut().take(19) {
                *r = None; // BL / BLR: caller-saved registers
            }
            page[30] = None;
        } else if w & 0x0a00_0000 == 0x0800_0000 {
            // Loads and stores: forget Rt, Rt2 and (with writeback) Rn.
            let pair = w & 0x3a00_0000 == 0x2800_0000;
            page[rd] = None;
            if pair {
                page[rt2] = None;
            }
            let writeback = if pair {
                w & 0x0080_0000 != 0
            } else {
                w & 0x0120_0400 == 0x0000_0400
            };
            if writeback {
                page[rn] = None;
            }
        } else {
            page[rd] = None;
        }
    }
    scanned
}

fn is_c_string_start(d: &[u8], s: &Section, value: u64) -> bool {
    let at = (s.offset + (value - s.address)) as usize;
    let end = (s.offset + s.size) as usize;
    if value != s.address && d[at - 1] != 0 {
        return false;
    }
    let mut n = 0;
    for b in &d[at..end] {
        match *b {
            0 => return n > 0,
            0x20..=0x7e | b'\t' | b'\n' | b'\r' => n += 1,
            _ => return false,
        }
    }
    false
}

/// Classify every word the rebase rule changes. `d` is the whole ELF file.
pub fn audit(d: &[u8]) -> Result<Audit, String> {
    let secs = sections(d)?;
    let (lo, hi) = image_bounds(d)?;
    let by_name = |name: &str| secs.iter().find(|s| s.name == name);
    let section_of = |va: u64| {
        secs.iter().find(|s| {
            s.flags & 2 != 0
                && s.size > 0
                && s.address <= va
                && va < s.address + s.size
                && !(s.kind == 8 && s.flags & 0x400 != 0)
        })
    };
    let mut a = Audit {
        image_lo: lo,
        image_hi: hi,
        ..Audit::default()
    };

    let mut functions =
        function_starts(d, by_name(".eh_frame_hdr").ok_or(".eh_frame_hdr missing")?)?;
    for name in [".init", ".fini"] {
        if let Some(s) = by_name(name) {
            functions.insert(s.address);
        }
    }
    let plt = by_name(".plt").cloned();
    let mut code_targets = HashSet::new();
    for s in secs
        .iter()
        .filter(|s| s.flags & 4 != 0 && s.kind == 1 && s.size > 0)
    {
        let bytes = &d[s.offset as usize..(s.offset + s.size) as usize];
        a.instructions_scanned += code_address_targets(bytes, s.address, &mut code_targets);
    }
    a.evidence_function_starts = functions.len() as u64;
    a.evidence_code_address_targets = code_targets.len() as u64;

    // Pass 1: the rebased set.
    let mut rebased: Vec<(usize, u64, u64)> = Vec::new(); // (rule section index, word address, value)
    for (index, name) in REBASE_SECTIONS.iter().enumerate() {
        let s = by_name(name).ok_or_else(|| format!("{name} missing"))?;
        if s.kind == 8 {
            return Err(format!("{name} has no file bytes"));
        }
        let skip = s.address.wrapping_neg() % 8;
        for k in 0..s.size.saturating_sub(skip) / 8 {
            let v = u64_at(d, s.offset + skip + 8 * k)?;
            if (lo..hi).contains(&v) {
                rebased.push((index, s.address + skip + 8 * k, v));
            }
        }
        // In-range values the rule cannot see because they are not 8-aligned.
        // (4-aligned but not 8-aligned offsets; AAPCS64 aligns pointers to 8.)
        for off in (0..s.size.saturating_sub(7)).filter(|off| (s.address + off) % 8 == 4) {
            if (lo..hi).contains(&u64_at(d, s.offset + off)?) {
                a.unaligned_in_range_in_rule_sections += 1;
            }
        }
    }
    a.candidates_total = rebased.len() as u64;
    let is_function = |v: u64| {
        functions.contains(&v)
            || plt.as_ref().is_some_and(|p| {
                v >= p.address && v < p.address + p.size && (v - p.address) % 16 == 0
            })
    };
    // Contradicted words leave the set (pointer-only sections are never second-guessed).
    rebased.retain(|(index, address, v)| {
        if POINTER_ONLY.contains(&REBASE_SECTIONS[*index]) {
            return true;
        }
        let reason = match section_of(*v) {
            None => "outside_sections",
            Some(s) if UNWIND.contains(&s.name.as_str()) => "unwind_metadata",
            Some(s) if s.flags & 4 != 0 && v % 4 != 0 => "executable_unaligned",
            Some(s) if s.flags & 4 != 0 && !is_function(*v) => "executable_not_function_start",
            Some(_) => return true,
        };
        *a.contradicted_by_reason.entry(reason).or_insert(0) += 1;
        if code_targets.contains(v) {
            a.contradicted_but_code_formed += 1;
        }
        a.contradicted.push((*address, *v));
        false
    });
    a.contradicted.sort_unstable();
    for reason in CONTRADICTIONS {
        a.contradicted_by_reason.entry(reason).or_insert(0);
    }
    let rebased_addresses: HashSet<u64> = rebased.iter().map(|r| r.1).collect();
    a.rebased_total = rebased.len() as u64;

    // Pass 2: classification.
    for (index, _address, v) in &rebased {
        let container = REBASE_SECTIONS[*index];
        *a.by_container.entry(container.to_string()).or_insert(0) += 1;
        let target = section_of(*v);
        let target_name = target.map_or("(no section)".to_string(), |s| s.name.clone());
        *a.by_target_section.entry(target_name.clone()).or_insert(0) += 1;
        let executable = target.is_some_and(|s| s.flags & 4 != 0);
        let is_function = is_function(*v);
        let is_code_target = code_targets.contains(v);
        if executable && !is_function {
            a.executable_target_not_function_start += 1;
        }
        let class = if POINTER_ONLY.contains(&container) {
            "pointer_only_section"
        } else if is_function {
            "function_start"
        } else if is_code_target {
            "code_address_target"
        } else if v % 8 == 0 && rebased_addresses.contains(v) {
            "pointer_bearing_object"
        } else if target.is_some_and(|s| s.name == ".rodata" && is_c_string_start(d, s, *v)) {
            "c_string_start"
        } else {
            a.ambiguous += 1;
            *a.ambiguous_by_container
                .entry(container.to_string())
                .or_insert(0) += 1;
            *a.ambiguous_by_target_section
                .entry(target_name)
                .or_insert(0) += 1;
            "ambiguous"
        };
        *a.by_class.entry(class).or_insert(0) += 1;
    }
    for class in CLASSES {
        a.by_class.entry(class).or_insert(0);
    }

    // The other direction: aligned in-range words the rule leaves alone.
    for s in secs.iter().filter(|s| {
        s.flags & 2 != 0
            && s.kind != 8
            && s.size >= 8
            && !REBASE_SECTIONS.contains(&s.name.as_str())
    }) {
        let skip = s.address.wrapping_neg() % 8;
        let mut count = 0u64;
        for k in 0..s.size.saturating_sub(skip) / 8 {
            if (lo..hi).contains(&u64_at(d, s.offset + skip + 8 * k)?) {
                count += 1;
            }
        }
        if count > 0 {
            a.not_rebased_in_range.insert(s.name.clone(), count);
        }
    }
    Ok(a)
}

fn json_map<K: AsRef<str>>(m: &BTreeMap<K, u64>) -> String {
    // Section and class names are plain ASCII identifiers; anything else is escaped.
    let quote = |s: &str| {
        let mut out = String::from("\"");
        for c in s.chars() {
            if c == '"' || c == '\\' || c.is_control() {
                out.push_str(&format!("\\u{:04x}", c as u32));
            } else {
                out.push(c);
            }
        }
        out.push('"');
        out
    };
    format!(
        "{{{}}}",
        m.iter()
            .map(|(k, v)| format!("{}:{v}", quote(k.as_ref())))
            .collect::<Vec<_>>()
            .join(",")
    )
}

impl Audit {
    pub fn to_json(&self, elf_sha256: &str) -> String {
        format!(
            "{{\"schema\":\"cuda-oxide.tileiras-native-rebase-audit.v1\",\"elf_sha256\":\"{elf_sha256}\",\"image_lo\":{},\"image_hi\":{},\"candidates_total\":{},\"contradicted_left_unslid\":{},\"contradicted_by_reason\":{},\"contradicted_but_code_formed\":{},\"rebased_total\":{},\"by_container\":{},\"by_class\":{},\"by_target_section\":{},\"ambiguous\":{},\"ambiguous_by_container\":{},\"ambiguous_by_target_section\":{},\"executable_target_not_function_start\":{},\"not_rebased_in_range\":{},\"unaligned_in_range_in_rule_sections\":{},\"evidence_function_starts\":{},\"evidence_code_address_targets\":{},\"instructions_scanned\":{},\"nvidia_code_executed\":false}}",
            self.image_lo, self.image_hi, self.candidates_total, self.contradicted.len(),
            json_map(&self.contradicted_by_reason), self.contradicted_but_code_formed, self.rebased_total, json_map(&self.by_container), json_map(&self.by_class),
            json_map(&self.by_target_section), self.ambiguous, json_map(&self.ambiguous_by_container),
            json_map(&self.ambiguous_by_target_section), self.executable_target_not_function_start,
            json_map(&self.not_rebased_in_range), self.unaligned_in_range_in_rule_sections, self.evidence_function_starts,
            self.evidence_code_address_targets, self.instructions_scanned
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn adrp_add_and_adr_targets_are_recorded() {
        let mut out = HashSet::new();
        // adrp x0, +0x3000 ; add x1, x0, #0x18 ; adr x2, +8
        // (adrp: immlo = 3 in bits 30:29, immhi = 0 -> 3 pages)
        let code = words(&[0x9000_0000 | (3 << 29), 0x9100_6001, 0x1000_0042]);
        assert_eq!(code_address_targets(&code, 0x40_1000, &mut out), 3);
        assert!(out.contains(&(0x40_4000 + 0x18)), "{out:x?}");
        assert!(out.contains(&(0x40_1008 + 8)), "{out:x?}");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_page_is_forgotten_after_a_write_a_call_or_a_branch() {
        for clobber in [
            0xd280_0000u32, // mov x0, #0
            0xf940_0020,    // ldr x0, [x1]
            0xa940_0020,    // ldp x0, x0, [x1]
            0x9400_0010,    // bl
            0x1400_0010,    // b
            0xd65f_03c0,    // ret
        ] {
            let mut out = HashSet::new();
            let code = words(&[0x9000_0000 | (1 << 29), clobber, 0x9100_6001]);
            code_address_targets(&code, 0x40_1000, &mut out);
            assert!(out.is_empty(), "{clobber:#x}: {out:x?}");
        }
        // A store through the register keeps it (no writeback) ...
        let mut out = HashSet::new();
        code_address_targets(
            &words(&[0x9000_0000 | (1 << 29), 0xf900_0002, 0x9100_6001]),
            0x40_1000,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        // ... a post-indexed load through it does not.
        let mut out = HashSet::new();
        code_address_targets(
            &words(&[0x9000_0000 | (1 << 29), 0xf840_8402, 0x9100_6001]),
            0x40_1000,
            &mut out,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn section_table_is_bounds_checked() {
        assert!(sections(b"\x7fELF\x02\x01").is_err());
        let mut d = vec![0u8; 64];
        d[..6].copy_from_slice(b"\x7fELF\x02\x01");
        d[58] = 64; // e_shentsize
        d[60] = 2; // e_shnum
        d[62] = 1; // e_shstrndx
        d[40] = 0x40; // e_shoff: points past the end
        assert!(sections(&d).is_err());
    }
}
