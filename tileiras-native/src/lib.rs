// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Non-executing Linux/AArch64 ET_EXEC admission and image planning.
//!
//! This milestone does not map executable memory, resolve imports, invoke
//! constructors, install a Linux thread pointer, or run NVIDIA code. A valid
//! plan is not an execution capability. The working PTX bridge is unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub mod a64;
pub mod patches;
pub mod rebase_audit;
pub mod sha256;
pub mod unwind_gate;

pub const MAX_FILE_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TABLE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SYMBOLS: u64 = 16_384;
const MAX_RELOCATIONS: usize = 262_144;
const MAX_TLS_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError(pub String);
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PlanError {}
type Result<T> = std::result::Result<T, PlanError>;
fn err(s: impl Into<String>) -> PlanError {
    PlanError(s.into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSegment {
    pub virtual_address: u64,
    pub file_offset: u64,
    pub file_size: u64,
    pub memory_size: u64,
    pub page_start: u64,
    pub page_end: u64,
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsTemplate {
    /// ELF template metadata, not an address to zero in the process image.
    pub virtual_address: u64,
    pub alignment: u64,
    pub memory_size: u64,
    pub initialized_bytes: Vec<u8>,
}
impl TlsTemplate {
    /// A separate per-thread byte image. No thread-pointer installation occurs.
    /// The caller must honor `alignment` when allocating a native TLS block;
    /// a Vec<u8> alone does not promise that native alignment.
    pub fn instantiate(&self) -> Result<Vec<u8>> {
        if self.memory_size > MAX_TLS_BYTES
            || self.initialized_bytes.len() as u64 > self.memory_size
        {
            return Err(err("invalid TLS template size"));
        }
        if self.alignment == 0
            || !self.alignment.is_power_of_two()
            || self.alignment > MAX_TLS_BYTES
        {
            return Err(err("invalid TLS alignment"));
        }
        let mut bytes = vec![0; self.memory_size as usize];
        bytes[..self.initialized_bytes.len()].copy_from_slice(&self.initialized_bytes);
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupInfo {
    pub interpreter: String,
    pub program_headers_address: u64,
    pub program_header_count: u16,
    pub init: Option<u64>,
    pub fini: Option<u64>,
    pub init_array: Vec<u64>,
    pub fini_array: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub name: String,
    pub binding: u8,
    pub symbol_type: u8,
    pub weak: bool,
    pub symbol_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relocation {
    pub kind: u32,
    pub target: u64,
    pub symbol_index: u32,
    pub symbol_name: String,
    pub addend: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelroRange {
    pub virtual_address: u64,
    pub size: u64,
    /// Outward-rounded pages touched by this range, NOT a protection request.
    pub page_start: u64,
    pub page_end: u64,
    /// Inward-rounded fully covered pages. Equal bounds denote an empty range.
    pub whole_page_start: u64,
    pub whole_page_end: u64,
    /// Both original boundaries are page-aligned. False requires a separate
    /// protection policy before execution; rounding outward may freeze data.
    pub exact_page_protection: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlan {
    pub input_sha256: String,
    pub file_size: u64,
    pub entry: u64,
    /// Outward-rounded mapping bounds, not just the logical PT_LOAD extent.
    pub load_base: u64,
    pub load_end: u64,
    pub host_page_size: u64,
    pub segments: Vec<LoadSegment>,
    pub tls: Option<TlsTemplate>,
    pub startup: StartupInfo,
    pub imports: Vec<Import>,
    pub relocations: Vec<Relocation>,
    pub needed_libraries: Vec<String>,
    pub relro: Vec<RelroRange>,
}
impl LoadPlan {
    /// Explicitly false even for the genuine pinned NVIDIA binary.
    pub fn execution_ready(&self) -> bool {
        false
    }

    /// Produce a non-executable host byte buffer, indexed from `load_base`.
    /// Re-admit the exact input and structural plan; no foreign calls occur.
    /// Only PT_LOAD tails are zero-filled. PT_TLS is never written here.
    pub fn materialize(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        if bytes.len() as u64 != self.file_size || sha256::hex(bytes) != self.input_sha256 {
            return Err(err("input bytes differ from admitted SHA256"));
        }
        if plan(bytes, self.host_page_size)? != *self {
            return Err(err("load plan differs from revalidated input"));
        }
        let mut image = vec![0; (self.load_end - self.load_base) as usize];
        for segment in &self.segments {
            let start = (segment.virtual_address - self.load_base) as usize;
            let raw = span(
                bytes,
                segment.file_offset,
                segment.file_size,
                "PT_LOAD bytes",
            )?;
            image[start..start + raw.len()].copy_from_slice(raw);
        }
        Ok(image)
    }
}

#[derive(Debug, Clone)]
struct Program {
    kind: u32,
    flags: u32,
    offset: u64,
    address: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}
#[derive(Debug, Clone)]
struct Section {
    kind: u32,
    address: u64,
    offset: u64,
    size: u64,
    link: u32,
    entsize: u64,
}

fn span<'a>(data: &'a [u8], offset: u64, size: u64, label: &str) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| err(format!("overflow in {label}")))?;
    if end > data.len() as u64 {
        return Err(err(format!("out-of-file {label}")));
    }
    Ok(&data[offset as usize..end as usize])
}
fn u16_at(data: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        span(data, offset as u64, 2, "u16")?.try_into().unwrap(),
    ))
}
fn u32_at(data: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        span(data, offset as u64, 4, "u32")?.try_into().unwrap(),
    ))
}
fn u64_at(data: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        span(data, offset as u64, 8, "u64")?.try_into().unwrap(),
    ))
}
fn end(start: u64, size: u64) -> Result<u64> {
    start
        .checked_add(size)
        .ok_or_else(|| err("address overflow"))
}
fn round_up(value: u64, align: u64) -> Result<u64> {
    Ok(end(value, align - 1)? & !(align - 1))
}
fn cstring(data: &[u8], offset: u64, label: &str) -> Result<String> {
    let tail = span(
        data,
        offset,
        (data.len() as u64)
            .checked_sub(offset)
            .ok_or_else(|| err("string offset out of range"))?,
        label,
    )?;
    let n = tail
        .iter()
        .take(4097)
        .position(|b| *b == 0)
        .ok_or_else(|| err(format!("unterminated/oversized {label}")))?;
    if n > 4096 || !tail[..n].is_ascii() {
        return Err(err(format!("non-ASCII/oversized {label}")));
    }
    Ok(String::from_utf8(tail[..n].to_vec()).unwrap())
}
fn va_offset(loads: &[LoadSegment], address: u64, size: u64) -> Result<u64> {
    let last = end(address, size)?;
    let matches: Vec<_> = loads
        .iter()
        .filter(|s| s.virtual_address <= address && last <= s.virtual_address + s.file_size)
        .collect();
    if matches.len() != 1 {
        return Err(err("virtual range is not uniquely file-backed"));
    }
    end(matches[0].file_offset, address - matches[0].virtual_address)
}
fn va_bytes<'a>(
    data: &'a [u8],
    loads: &[LoadSegment],
    address: u64,
    size: u64,
) -> Result<&'a [u8]> {
    if size > MAX_TABLE_BYTES {
        return Err(err("oversized virtual table"));
    }
    span(
        data,
        va_offset(loads, address, size)?,
        size,
        "virtual table",
    )
}
fn executable(loads: &[LoadSegment], address: u64) -> bool {
    address.is_multiple_of(4)
        && loads.iter().any(|s| {
            s.execute
                && s.virtual_address <= address
                && address
                    .checked_add(4)
                    .is_some_and(|last| last <= s.virtual_address + s.file_size)
        })
}
fn single(tags: &BTreeMap<u64, Vec<u64>>, tag: u64) -> Result<u64> {
    match tags.get(&tag).map(Vec::as_slice) {
        Some([value]) => Ok(*value),
        _ => Err(err(format!("missing/duplicate dynamic tag {tag}"))),
    }
}
fn optional(tags: &BTreeMap<u64, Vec<u64>>, tag: u64) -> Result<Option<u64>> {
    if tags.contains_key(&tag) {
        Ok(Some(single(tags, tag)?))
    } else {
        Ok(None)
    }
}

fn relro_plan(programs: &[Program], loads: &[LoadSegment], page: u64) -> Result<Vec<RelroRange>> {
    let headers: Vec<_> = programs.iter().filter(|p| p.kind == 0x6474e552).collect();
    if headers.len() > 1 {
        return Err(err("duplicate PT_GNU_RELRO"));
    }
    let mut ranges = Vec::new();
    for p in headers {
        let limit = end(p.address, p.memsz)?;
        if p.memsz == 0
            || p.filesz != p.memsz
            || !loads.iter().any(|s| {
                s.write
                    && s.virtual_address <= p.address
                    && limit <= s.virtual_address + s.memory_size
            })
        {
            return Err(err("PT_GNU_RELRO outside supported writable image layout"));
        }
        if va_offset(loads, p.address, p.filesz)? != p.offset {
            return Err(err("PT_GNU_RELRO file mapping mismatch"));
        }
        let whole_page_end = limit & !(page - 1);
        ranges.push(RelroRange {
            virtual_address: p.address,
            size: p.memsz,
            page_start: p.address & !(page - 1),
            page_end: round_up(limit, page)?,
            whole_page_start: round_up(p.address, page)?.min(whole_page_end),
            whole_page_end,
            exact_page_protection: p.address % page == 0 && limit % page == 0,
        });
    }
    Ok(ranges)
}

fn validate_phdr(
    programs: &[Program],
    file_offset: u64,
    mapped_address: u64,
    table_size: u64,
) -> Result<()> {
    let headers: Vec<_> = programs.iter().filter(|p| p.kind == 6).collect();
    if headers.len() > 1 {
        return Err(err("duplicate PT_PHDR"));
    }
    if let Some(p) = headers.first() {
        if p.offset != file_offset
            || p.address != mapped_address
            || p.filesz != table_size
            || p.memsz != table_size
            || p.flags != 4
        {
            return Err(err("PT_PHDR contradicts mapped program table"));
        }
    }
    Ok(())
}

/// Build a bounded ET_EXEC image plan without mapping or executing any bytes.
pub fn plan(data: &[u8], host_page_size: u64) -> Result<LoadPlan> {
    if data.len() < 64 || data.len() > MAX_FILE_BYTES {
        return Err(err("ELF input size outside bounds"));
    }
    if !(4096..=65536).contains(&host_page_size) || !host_page_size.is_power_of_two() {
        return Err(err("unsupported host page size"));
    }
    if &data[..7] != b"\x7fELF\x02\x01\x01" || ![0, 3].contains(&data[7]) || data[8] != 0 {
        return Err(err("expected ELF64 little-endian SystemV/GNU version 1"));
    }
    if u16_at(data, 16)? != 2 || u16_at(data, 18)? != 183 || u32_at(data, 20)? != 1 {
        return Err(err("only AArch64 ET_EXEC is admitted"));
    }
    if u16_at(data, 52)? != 64
        || u16_at(data, 54)? != 56
        || u16_at(data, 58)? != 64
        || u32_at(data, 48)? != 0
    {
        return Err(err("unsupported ELF header shape/flags"));
    }
    let (entry, phoff, shoff) = (u64_at(data, 24)?, u64_at(data, 32)?, u64_at(data, 40)?);
    let (phnum, shnum, shstrndx) = (u16_at(data, 56)?, u16_at(data, 60)?, u16_at(data, 62)?);
    if !(1..=1024).contains(&phnum)
        || !(1..=4096).contains(&shnum)
        || shstrndx == 0
        || shstrndx >= shnum
    {
        return Err(err("unsupported ELF table count/extended numbering"));
    }
    let phbytes = span(data, phoff, phnum as u64 * 56, "program table")?;
    let shbytes = span(data, shoff, shnum as u64 * 64, "section table")?;
    let mut programs = Vec::new();
    let mut segments = Vec::new();
    for row in phbytes.chunks_exact(56) {
        let p = Program {
            kind: u32_at(row, 0)?,
            flags: u32_at(row, 4)?,
            offset: u64_at(row, 8)?,
            address: u64_at(row, 16)?,
            filesz: u64_at(row, 32)?,
            memsz: u64_at(row, 40)?,
            align: u64_at(row, 48)?,
        };
        span(data, p.offset, p.filesz, "program contents")?;
        end(p.address, p.memsz)?;
        if p.align > 1 && !p.align.is_power_of_two() {
            return Err(err("invalid program alignment"));
        }
        if p.flags & !7 != 0 {
            return Err(err("unsupported program flags"));
        }
        if [1, 7].contains(&p.kind) && p.memsz < p.filesz {
            return Err(err("program memory smaller than file data"));
        }
        if p.kind == 1 {
            if p.memsz == 0 || p.memsz > MAX_IMAGE_BYTES || p.flags & 4 == 0 || p.flags & 3 == 3 {
                return Err(err("unsupported empty/oversized/unreadable/W+X PT_LOAD"));
            }
            if p.align > 1 && p.address % p.align != p.offset % p.align {
                return Err(err("incongruent PT_LOAD alignment"));
            }
            let page_start = p.address & !(host_page_size - 1);
            let page_end = round_up(end(p.address, p.memsz)?, host_page_size)?;
            if page_start < host_page_size {
                return Err(err("PT_LOAD overlaps null-page guard"));
            }
            segments.push(LoadSegment {
                virtual_address: p.address,
                file_offset: p.offset,
                file_size: p.filesz,
                memory_size: p.memsz,
                page_start,
                page_end,
                read: p.flags & 4 != 0,
                write: p.flags & 2 != 0,
                execute: p.flags & 1 != 0,
            });
        }
        if p.kind == 0x6474e551 && p.flags & 1 != 0 {
            return Err(err("executable GNU stack refused"));
        }
        programs.push(p);
    }
    segments.sort_by_key(|s| s.virtual_address);
    if segments.is_empty() {
        return Err(err("no PT_LOAD image"));
    }
    for pair in segments.windows(2) {
        if pair[0].page_end > pair[1].page_start {
            return Err(err("PT_LOAD memory/pages overlap"));
        }
    }
    let load_base = segments[0].page_start;
    let load_end = segments.last().unwrap().page_end;
    if load_end - load_base > MAX_IMAGE_BYTES {
        return Err(err("image span exceeds bound"));
    }
    if !executable(&segments, entry) {
        return Err(err("entry is outside file-backed executable code"));
    }
    let ph_loads: Vec<_> = segments
        .iter()
        .filter(|s| {
            s.file_offset <= phoff && phoff + phbytes.len() as u64 <= s.file_offset + s.file_size
        })
        .collect();
    if ph_loads.len() != 1 {
        return Err(err("program table has no unique PT_LOAD mapping"));
    }
    let program_headers_address = ph_loads[0].virtual_address + phoff - ph_loads[0].file_offset;
    validate_phdr(
        &programs,
        phoff,
        program_headers_address,
        phbytes.len() as u64,
    )?;
    let interps: Vec<_> = programs.iter().filter(|p| p.kind == 3).collect();
    if interps.len() != 1 || interps[0].filesz > 4096 {
        return Err(err("expected one bounded interpreter"));
    }
    let interp = span(data, interps[0].offset, interps[0].filesz, "interpreter")?;
    let interpreter = cstring(interp, 0, "interpreter")?;
    if interpreter != "/lib/ld-linux-aarch64.so.1" || interpreter.len() + 1 != interp.len() {
        return Err(err("unsupported Linux interpreter"));
    }

    let mut sections = Vec::new();
    let mut name_offsets = Vec::new();
    for row in shbytes.chunks_exact(64) {
        name_offsets.push(u32_at(row, 0)?);
        let s = Section {
            kind: u32_at(row, 4)?,
            address: u64_at(row, 16)?,
            offset: u64_at(row, 24)?,
            size: u64_at(row, 32)?,
            link: u32_at(row, 40)?,
            entsize: u64_at(row, 56)?,
        };
        if s.kind != 8 {
            span(data, s.offset, s.size, "section contents")?;
        }
        end(s.address, s.size)?;
        let alignment = u64_at(row, 48)?;
        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(err("invalid section alignment"));
        }
        if [9, 19].contains(&s.kind) {
            return Err(err("REL/RELR sections not supported"));
        }
        sections.push(s);
    }
    let names = &sections[shstrndx as usize];
    if names.kind != 3 || names.size > MAX_TABLE_BYTES {
        return Err(err("invalid section names"));
    }
    let names_raw = span(data, names.offset, names.size, "section names")?;
    for off in name_offsets {
        cstring(names_raw, off as u64, "section name")?;
    }
    let sym_indices: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind == 11)
        .map(|(i, _)| i)
        .collect();
    if sym_indices.len() != 1 {
        return Err(err("expected one dynamic symbol table"));
    }
    let sym_index = sym_indices[0];
    let sym = &sections[sym_index];
    if sym.entsize != 24 || sym.size == 0 || sym.size % 24 != 0 || sym.size / 24 > MAX_SYMBOLS {
        return Err(err("invalid dynamic symbol table shape"));
    }
    let strings = sections
        .get(sym.link as usize)
        .ok_or_else(|| err("bad dynsym string link"))?;
    if strings.kind != 3 || strings.size > MAX_TABLE_BYTES {
        return Err(err("invalid dynamic strings"));
    }
    if va_offset(&segments, sym.address, sym.size)? != sym.offset
        || va_offset(&segments, strings.address, strings.size)? != strings.offset
    {
        return Err(err("symbol/string section mapping differs from PT_LOAD"));
    }
    let strdata = span(data, strings.offset, strings.size, "dynamic strings")?;
    if strdata.first() != Some(&0) {
        return Err(err("dynamic string table lacks empty index zero"));
    }
    let mut imports = Vec::new();
    let mut symbol_names = Vec::new();
    for (index, row) in span(data, sym.offset, sym.size, "dynsym")?
        .chunks_exact(24)
        .enumerate()
    {
        let name = cstring(strdata, u32_at(row, 0)? as u64, "dynamic symbol")?;
        let shndx = u16_at(row, 6)?;
        if index == 0 && row.iter().any(|b| *b != 0) {
            return Err(err("invalid null dynamic symbol"));
        }
        if shndx == 0 && !name.is_empty() {
            let binding = row[4] >> 4;
            if ![1, 2].contains(&binding) {
                return Err(err("unsupported undefined symbol binding"));
            }
            imports.push(Import {
                name: name.clone(),
                binding,
                symbol_type: row[4] & 15,
                weak: binding == 2,
                symbol_index: index as u32,
            });
        }
        symbol_names.push(name);
    }

    let dynamics: Vec<_> = programs.iter().filter(|p| p.kind == 2).collect();
    if dynamics.len() != 1 {
        return Err(err("expected one PT_DYNAMIC"));
    }
    let dynamic = dynamics[0];
    if dynamic.filesz == 0 || dynamic.filesz % 16 != 0 || dynamic.filesz > 16 * 4096 {
        return Err(err("invalid dynamic table size"));
    }
    if va_offset(&segments, dynamic.address, dynamic.filesz)? != dynamic.offset {
        return Err(err("PT_DYNAMIC file mapping mismatch"));
    }
    let mut tags: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut terminated = false;
    for row in span(data, dynamic.offset, dynamic.filesz, "dynamic table")?.chunks_exact(16) {
        let (tag, value) = (u64_at(row, 0)?, u64_at(row, 8)?);
        if tag == 0 {
            terminated = true;
            break;
        }
        tags.entry(tag).or_default().push(value);
    }
    if !terminated {
        return Err(err("unterminated dynamic table"));
    }
    if single(&tags, 5)? != strings.address
        || single(&tags, 10)? != strings.size
        || single(&tags, 6)? != sym.address
        || single(&tags, 11)? != 24
    {
        return Err(err("dynamic symbol/string tags disagree with sections"));
    }
    if [17, 18, 19, 35, 36, 37]
        .iter()
        .any(|tag| tags.contains_key(tag))
    {
        return Err(err("REL/RELR dynamic tables not supported"));
    }
    let needed = tags
        .get(&1)
        .ok_or_else(|| err("missing shared-library dependencies"))?;
    if needed.is_empty() || needed.len() > 256 {
        return Err(err("invalid dependency count"));
    }
    let mut needed_libraries = Vec::new();
    for off in needed {
        let name = cstring(strdata, *off, "DT_NEEDED")?;
        if name.is_empty() || name.contains('/') {
            return Err(err("invalid dependency basename"));
        }
        needed_libraries.push(name);
    }
    let rela_sections: Vec<_> = sections.iter().filter(|s| s.kind == 4).collect();
    let mut relocation_ranges = BTreeSet::new();
    let mut relocation_targets = BTreeSet::new();
    let mut relocations = Vec::new();
    for s in &rela_sections {
        if s.entsize != 24
            || s.size % 24 != 0
            || s.size > MAX_TABLE_BYTES
            || s.link as usize != sym_index
        {
            return Err(err("invalid RELA section shape/link"));
        }
        if va_offset(&segments, s.address, s.size)? != s.offset {
            return Err(err("RELA section file mapping mismatch"));
        }
        if !relocation_ranges.insert((s.address, s.size)) {
            return Err(err("duplicate relocation range"));
        }
        for row in span(data, s.offset, s.size, "RELA contents")?.chunks_exact(24) {
            if relocations.len() >= MAX_RELOCATIONS {
                return Err(err("too many relocations"));
            }
            let (target, info, addend) =
                (u64_at(row, 0)?, u64_at(row, 8)?, u64_at(row, 16)? as i64);
            let (kind, symbol_index) = (info as u32, (info >> 32) as u32);
            if ![1025, 1026].contains(&kind) {
                return Err(err(format!("unsupported AArch64 relocation {kind}")));
            }
            let symbol_name = symbol_names
                .get(symbol_index as usize)
                .ok_or_else(|| err("relocation symbol index out of range"))?
                .clone();
            if symbol_index == 0 || symbol_name.is_empty() {
                return Err(err("relocation has no named symbol"));
            }
            let target_end = end(target, 8)?;
            if target % 8 != 0
                || !segments.iter().any(|s| {
                    s.write
                        && s.virtual_address <= target
                        && target_end <= s.virtual_address + s.memory_size
                })
            {
                return Err(err(
                    "relocation destination is not aligned writable image memory",
                ));
            }
            if !relocation_targets.insert(target) {
                return Err(err("duplicate relocation target"));
            }
            relocations.push(Relocation {
                kind,
                target,
                symbol_index,
                symbol_name,
                addend,
            });
        }
    }
    let mut tagged_ranges = BTreeSet::new();
    for (address_tag, size_tag) in [(7, 8), (23, 2)] {
        if tags.contains_key(&address_tag) || tags.contains_key(&size_tag) {
            let range = (single(&tags, address_tag)?, single(&tags, size_tag)?);
            if !relocation_ranges.contains(&range) || !tagged_ranges.insert(range) {
                return Err(err("dynamic RELA range missing or duplicated"));
            }
            va_bytes(data, &segments, range.0, range.1)?;
        }
    }
    if tagged_ranges != relocation_ranges {
        return Err(err("untagged RELA section"));
    }
    if (tags.contains_key(&7) && single(&tags, 9)? != 24)
        || (tags.contains_key(&23) && single(&tags, 20)? != 7)
    {
        return Err(err("unsupported dynamic relocation encoding"));
    }
    let entry_option = |tag| -> Result<Option<u64>> {
        let value = optional(&tags, tag)?;
        if value.is_some_and(|addr| !executable(&segments, addr)) {
            return Err(err("init/fini target outside executable code"));
        }
        Ok(value)
    };
    let pointer_array = |address_tag, size_tag| -> Result<Vec<u64>> {
        if !tags.contains_key(&address_tag) && !tags.contains_key(&size_tag) {
            return Ok(Vec::new());
        }
        let (addr, size) = (single(&tags, address_tag)?, single(&tags, size_tag)?);
        if addr % 8 != 0 || size % 8 != 0 || size / 8 > 65536 {
            return Err(err("invalid constructor array shape"));
        }
        let mut pointers = Vec::new();
        for row in va_bytes(data, &segments, addr, size)?.chunks_exact(8) {
            let target = u64_at(row, 0)?;
            if !executable(&segments, target) {
                return Err(err("constructor/destructor target outside executable code"));
            }
            pointers.push(target);
        }
        Ok(pointers)
    };
    let startup = StartupInfo {
        interpreter,
        program_headers_address,
        program_header_count: phnum,
        init: entry_option(12)?,
        fini: entry_option(13)?,
        init_array: pointer_array(25, 27)?,
        fini_array: pointer_array(26, 28)?,
    };
    let tls_headers: Vec<_> = programs.iter().filter(|p| p.kind == 7).collect();
    if tls_headers.len() > 1 {
        return Err(err("duplicate PT_TLS"));
    }
    let tls = if let Some(t) = tls_headers.first() {
        if t.memsz > MAX_TLS_BYTES
            || t.align > MAX_TLS_BYTES
            || !segments.iter().any(|s| {
                s.virtual_address <= t.address
                    && t.address + t.memsz <= s.virtual_address + s.memory_size
            })
        {
            return Err(err("PT_TLS outside supported image layout/size bound"));
        }
        if t.filesz != 0 && va_offset(&segments, t.address, t.filesz)? != t.offset {
            return Err(err("initialized TLS template mapping mismatch"));
        }
        Some(TlsTemplate {
            virtual_address: t.address,
            alignment: t.align.max(1),
            memory_size: t.memsz,
            initialized_bytes: span(data, t.offset, t.filesz, "TLS template")?.to_vec(),
        })
    } else {
        None
    };
    let relro = relro_plan(&programs, &segments, host_page_size)?;
    Ok(LoadPlan {
        input_sha256: sha256::hex(data),
        file_size: data.len() as u64,
        entry,
        load_base,
        load_end,
        host_page_size,
        segments,
        tls,
        startup,
        imports,
        relocations,
        needed_libraries,
        relro,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn writable_segment() -> LoadSegment {
        LoadSegment {
            virtual_address: 0x10000,
            file_offset: 0,
            file_size: 0x10000,
            memory_size: 0x10000,
            page_start: 0x10000,
            page_end: 0x20000,
            read: true,
            write: true,
            execute: false,
        }
    }

    fn relro(address: u64, size: u64) -> Program {
        Program {
            kind: 0x6474e552,
            flags: 4,
            offset: address - 0x10000,
            address,
            filesz: size,
            memsz: size,
            align: 1,
        }
    }

    #[test]
    fn relro_partial_pages_are_described_not_promised() {
        let p = relro(0x10218, 0x5de8);
        let r = relro_plan(&[p], &[writable_segment()], 0x4000)
            .unwrap()
            .remove(0);
        assert_eq!((r.virtual_address, r.size), (0x10218, 0x5de8));
        assert_eq!((r.page_start, r.page_end), (0x10000, 0x18000));
        assert_eq!((r.whole_page_start, r.whole_page_end), (0x14000, 0x14000));
        assert!(!r.exact_page_protection);
    }

    #[test]
    fn relro_exact_page_bounds_are_distinct_from_runtime_admission() {
        let r = relro_plan(&[relro(0x10000, 0x8000)], &[writable_segment()], 0x4000)
            .unwrap()
            .remove(0);
        assert!(r.exact_page_protection);
        assert_eq!((r.whole_page_start, r.whole_page_end), (0x10000, 0x18000));
    }

    #[test]
    fn relro_subpage_has_no_complete_page() {
        let r = relro_plan(&[relro(0x10218, 16)], &[writable_segment()], 0x4000)
            .unwrap()
            .remove(0);
        assert_eq!(r.whole_page_start, r.whole_page_end);
        assert!(!r.exact_page_protection);
    }

    #[test]
    fn relro_duplicates_bounds_and_backing_are_validated() {
        let r = relro(0x10000, 0x4000);
        assert!(relro_plan(&[r.clone(), r], &[writable_segment()], 0x4000).is_err());
        assert!(relro_plan(&[relro(0x20000, 0x4000)], &[writable_segment()], 0x4000).is_err());
        let mut r = relro(0x10000, 0x4000);
        r.offset = 8;
        assert!(relro_plan(&[r], &[writable_segment()], 0x4000).is_err());
        let mut load = writable_segment();
        load.write = false;
        assert!(relro_plan(&[relro(0x10000, 0x4000)], &[load], 0x4000).is_err());
    }

    #[test]
    fn tls_instances_are_private_template_copies_not_thread_pointers() {
        let t = TlsTemplate {
            virtual_address: 0x10000,
            alignment: 8,
            memory_size: 12,
            initialized_bytes: vec![1, 2, 3, 4],
        };
        let mut a = t.instantiate().unwrap();
        let b = t.instantiate().unwrap();
        assert_eq!(b, [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
        a[0] = 99;
        assert_eq!(b[0], 1);
        assert_eq!(t.initialized_bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn tls_template_revalidates_public_fields_before_allocation() {
        let mut t = TlsTemplate {
            virtual_address: 0,
            alignment: 8,
            memory_size: 4,
            initialized_bytes: vec![0; 8],
        };
        assert!(t.instantiate().is_err());
        t.memory_size = MAX_TLS_BYTES + 1;
        assert!(t.instantiate().is_err());
        t.memory_size = 8;
        for bad in [0, 3, MAX_TLS_BYTES * 2] {
            t.alignment = bad;
            assert!(t.instantiate().is_err());
        }
    }

    #[test]
    fn arithmetic_file_and_string_bounds_fail_without_panics() {
        assert!(span(&[0; 8], u64::MAX, 8, "test").is_err());
        assert!(span(&[0; 8], 4, 5, "test").is_err());
        assert_eq!(span(&[0; 8], 8, 0, "test").unwrap(), []);
        assert!(round_up(u64::MAX, 16384).is_err());
        assert!(cstring(b"abc", 0, "test").is_err());
        assert!(cstring(&[0xff, 0], 0, "test").is_err());
        assert!(cstring(&[b'a'; 4098], 0, "test").is_err());
        assert!(cstring(b"\0", 2, "test").is_err());
    }

    #[test]
    fn overlapping_file_backing_is_ambiguous() {
        let load = writable_segment();
        assert!(va_offset(&[load.clone(), load.clone()], 0x10000, 8).is_err());
        assert_eq!(va_offset(&[load], 0x10100, 8).unwrap(), 0x100);
    }

    #[test]
    fn only_executable_file_bytes_can_be_called_in_a_future_loader() {
        let mut load = writable_segment();
        assert!(!executable(&[load.clone()], 0x10000));
        load.execute = true;
        load.file_size = 8;
        assert!(executable(&[load.clone()], 0x10004));
        assert!(!executable(&[load.clone()], 0x10008));
        assert!(!executable(&[load], 0x10001));
    }

    #[test]
    fn empty_truncated_and_incompatible_inputs_are_refused() {
        for size in [0, 1, 7, 63, 64, 128] {
            assert!(plan(&vec![0; size], 16384).is_err());
        }
        for page in [0, 1, 2048, 8193, 131072, u64::MAX] {
            assert!(plan(&[0; 64], page).is_err());
        }
    }

    #[test]
    fn explicit_phdr_must_describe_the_actual_mapped_table() {
        let p = Program {
            kind: 6,
            flags: 4,
            offset: 64,
            address: 0x400040,
            filesz: 560,
            memsz: 560,
            align: 8,
        };
        assert!(validate_phdr(&[], 64, 0x400040, 560).is_ok());
        assert!(validate_phdr(&[p.clone()], 64, 0x400040, 560).is_ok());
        assert!(validate_phdr(&[p.clone(), p.clone()], 64, 0x400040, 560).is_err());
        for changed in [
            Program {
                offset: 72,
                ..p.clone()
            },
            Program {
                address: 0x400048,
                ..p.clone()
            },
            Program {
                filesz: 568,
                ..p.clone()
            },
            Program {
                memsz: 568,
                ..p.clone()
            },
            Program {
                flags: 6,
                ..p.clone()
            },
        ] {
            assert!(validate_phdr(&[changed], 64, 0x400040, 560).is_err());
        }
    }
}
