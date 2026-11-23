// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: load one pinned Linux-aarch64 shared object
// (ET_DYN; NVIDIA's linux-sbsa libnvvm.so.4) next to the tileiras image, for
// the guest's dlopen("libnvvm.so").
//
// Unlike the ET_EXEC, a shared object carries real relocation records, so
// nothing here is heuristic: RELATIVE, ABS64, GLOB_DAT, JUMP_SLOT and TLSDESC
// are applied exactly; any other type is refused. Its x18 / TPIDR_EL0 sites
// come from a second SHA-bound patch manifest. TLS is static: the object's
// PT_TLS block lives at a fixed offset of every thread block and each TLSDESC
// slot gets a resolver stub returning that offset (`ldr x0, [x0, #8]; ret`).

use std::collections::BTreeMap;
use std::ffi::c_void;

use tileiras_native::a64;
use tileiras_native::patches::{Manifest, PatchKind};
use tileiras_native::sha256;

use crate::image::{self, Binding, PAGE};

/// Fixed base, 1 GiB above the executable's reservation start (which ends
/// well below it); refused if occupied, like the executable's range.
pub const BASE: u64 = image::SLIDE + 0x1000_0000;
const MAX_TABLE: u64 = 64 * 1024 * 1024;

pub struct Dso {
    pub base: u64,
    /// End of the object's own pages; the helper's islands follow up to `end`.
    pub image_end: u64,
    pub end: u64,
    pub phdr: u64,
    pub phnum: u64,
    pub init: Vec<u64>,
    pub exports: BTreeMap<String, u64>,
    pub tls_bytes: u64,
    pub relocations: BTreeMap<&'static str, u64>,
    pub imports_bound: usize,
    pub imports_trapping: Vec<String>,
    /// Unmapped trap address -> name of each unimplemented DATA import.
    pub data_traps: BTreeMap<u64, String>,
    pub x18_sites: usize,
    pub tls_sites: usize,
}

pub struct Inputs<'a> {
    pub elf: &'a [u8],
    pub elf_sha256: &'a str,
    pub manifest: &'a Manifest,
    pub tp_slot: u64,
    pub x18_slot: u64,
    /// Offset of this object's TLS block from the thread pointer.
    pub tls_offset: u64,
    pub tls_capacity: u64,
    pub resolve: &'a dyn Fn(&str) -> Option<Binding>,
    pub trap_handler: u64,
    pub errno_sync: u64,
    /// Index of this object's first import in the helper's trap-name table.
    pub trap_index_base: usize,
}

struct Load {
    va: u64,
    off: u64,
    filesz: u64,
    memsz: u64,
    flags: u32,
}

fn rd<const N: usize>(d: &[u8], at: u64) -> Result<[u8; N], String> {
    let at = usize::try_from(at).map_err(|_| "offset overflow")?;
    d.get(at..at.checked_add(N).ok_or("offset overflow")?)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| "read outside the shared object".to_string())
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

/// Parsed, validated view of the shared object (no memory is touched).
struct View<'a> {
    d: &'a [u8],
    loads: Vec<Load>,
    tags: BTreeMap<u64, u64>,
    tls: Option<(u64, u64, u64)>, // filesz, memsz, align
    relro: Option<(u64, u64)>,
    phoff: u64,
    phnum: u64,
}

impl<'a> View<'a> {
    fn parse(d: &'a [u8]) -> Result<Self, String> {
        if d.len() < 64 || &d[..7] != b"\x7fELF\x02\x01\x01" {
            return Err("not an ELF64 little-endian object".into());
        }
        if u16_at(d, 16)? != 3 || u16_at(d, 18)? != 183 || u16_at(d, 54)? != 56 {
            return Err("only an AArch64 ET_DYN is admitted".into());
        }
        let (phoff, phnum) = (u64_at(d, 32)?, u16_at(d, 56)? as u64);
        if phnum == 0 || phnum > 64 {
            return Err("unsupported program header count".into());
        }
        let (mut loads, mut dynamic, mut tls, mut relro) = (Vec::new(), None, None, None);
        for k in 0..phnum {
            let p = phoff + 56 * k;
            let (kind, flags) = (u32_at(d, p)?, u32_at(d, p + 4)?);
            let (off, va, filesz, memsz, align) = (
                u64_at(d, p + 8)?,
                u64_at(d, p + 16)?,
                u64_at(d, p + 32)?,
                u64_at(d, p + 40)?,
                u64_at(d, p + 48)?,
            );
            if off.checked_add(filesz).is_none_or(|e| e > d.len() as u64)
                || va.checked_add(memsz).is_none_or(|e| e > image::SLIDE)
                || memsz < filesz && kind == 1
            {
                return Err("program header outside the file or address space".into());
            }
            match kind {
                1 => loads.push(Load {
                    va,
                    off,
                    filesz,
                    memsz,
                    flags,
                }),
                2 => dynamic = Some((off, filesz)),
                7 => tls = Some((filesz, memsz, align)),
                0x6474_e552 => relro = Some((va, memsz)),
                _ => {}
            }
        }
        match &loads[..] {
            [t, w]
                if t.va == 0
                    && t.off == 0
                    && t.flags == 5
                    && w.flags == 6
                    && (t.va + t.memsz).div_ceil(PAGE) * PAGE <= w.va / PAGE * PAGE
                    && w.va % PAGE == w.off % PAGE
                    && phoff + 56 * phnum <= t.filesz => {}
            _ => {
                return Err(
                    "expected one R-X segment at 0 and one RW- segment on disjoint 16 KiB pages"
                        .into(),
                )
            }
        }
        let (doff, dsize) = dynamic.ok_or("no PT_DYNAMIC")?;
        let mut tags = BTreeMap::new();
        for k in 0..dsize / 16 {
            let (tag, value) = (u64_at(d, doff + 16 * k)?, u64_at(d, doff + 16 * k + 8)?);
            if tag == 0 {
                break;
            }
            if tag != 1 && tags.insert(tag, value).is_some() {
                return Err(format!("duplicate dynamic tag {tag}"));
            }
        }
        // REL / RELR / text relocations / symbolic binding are not implemented.
        for refused in [17u64, 18, 19, 22, 35, 36, 37] {
            if tags.contains_key(&refused) {
                return Err(format!("unsupported dynamic tag {refused}"));
            }
        }
        Ok(Self {
            d,
            loads,
            tags,
            tls,
            relro,
            phoff,
            phnum,
        })
    }

    /// File offset of `size` bytes at link address `va` (file-backed only).
    fn offset(&self, va: u64, size: u64) -> Result<u64, String> {
        self.loads
            .iter()
            .find(|l| l.va <= va && va.checked_add(size).is_some_and(|e| e <= l.va + l.filesz))
            .map(|l| l.off + (va - l.va))
            .ok_or_else(|| format!("address {va:#x} is not file-backed"))
    }
    fn tag(&self, tag: u64) -> Result<u64, String> {
        self.tags
            .get(&tag)
            .copied()
            .ok_or_else(|| format!("dynamic tag {tag} missing"))
    }
    fn string(&self, index: u64) -> Result<String, String> {
        let (strtab, strsz) = (self.tag(5)?, self.tag(10)?);
        if index >= strsz {
            return Err("string index out of range".into());
        }
        // The whole remainder of the string table must be file-backed; the
        // name must end inside it.
        let start = strtab.checked_add(index).ok_or("string table overflow")?;
        let off = self.offset(start, strsz - index)? as usize;
        let tail = &self.d[off..off + (strsz - index) as usize];
        let n = tail
            .iter()
            .take(1024)
            .position(|b| *b == 0)
            .ok_or("unterminated symbol name")?;
        String::from_utf8(tail[..n].to_vec()).map_err(|_| "non-UTF-8 symbol name".into())
    }
    /// (name, shndx, value, info) of dynamic symbol `index`.
    fn symbol(&self, index: u64, count: u64) -> Result<(String, u16, u64, u8), String> {
        if index >= count {
            return Err("symbol index out of range".into());
        }
        let at = index
            .checked_mul(24)
            .and_then(|o| self.tag(6).ok()?.checked_add(o))
            .ok_or("symbol table overflow")?;
        let off = self.offset(at, 24)?;
        Ok((
            self.string(u32_at(self.d, off)? as u64)?,
            u16_at(self.d, off + 6)?,
            u64_at(self.d, off + 8)?,
            self.d[off as usize + 4],
        ))
    }
}

pub fn load(i: &Inputs) -> Result<Dso, String> {
    let m = i.manifest;
    if sha256::hex(i.elf) != i.elf_sha256 || m.elf_sha256 != i.elf_sha256 {
        return Err("shared object / manifest SHA-256 mismatch".into());
    }
    let v = View::parse(i.elf)?;
    let (text, data) = (&v.loads[0], &v.loads[1]);
    let link_hi = data.va + data.memsz;
    if m.image_lo != 0 || m.image_hi != link_hi || !m.rebase.is_empty() {
        return Err("manifest does not describe this shared object".into());
    }
    let tls_bytes = match v.tls {
        None => 0,
        Some((0, memsz, align)) if memsz <= i.tls_capacity && align <= 16 => memsz,
        Some(_) => return Err("only a small zero-initialised PT_TLS is supported".into()),
    };
    // DT_HASH: nchain is the dynamic symbol count.
    let hash = v.offset(v.tag(4)?, 8)?;
    let nsyms = u32_at(v.d, hash + 4)? as u64;
    if v.tag(11)? != 24 || nsyms == 0 || nsyms > 1 << 20 {
        return Err("unsupported dynamic symbol table".into());
    }
    let mut tables = Vec::new();
    if let Some(&rela) = v.tags.get(&7) {
        if v.tag(9)? != 24 {
            return Err("unexpected DT_RELAENT".into());
        }
        tables.push((rela, v.tag(8)?));
    }
    if let Some(&jmprel) = v.tags.get(&23) {
        if v.tag(20)? != 7 {
            return Err("DT_PLTREL is not RELA".into());
        }
        tables.push((jmprel, v.tag(2)?));
    }
    let x18_sites = m.count(|k| matches!(k, PatchKind::X18(_)));
    let tls_sites = m.count(|k| *k == PatchKind::Tls);
    let undefined = (1..nsyms)
        .filter(|k| v.symbol(*k, nsyms).is_ok_and(|s| s.1 == 0))
        .count() as u64;
    let image_end = image::up(link_hi);
    let island_bytes = image::up(
        x18_sites as u64 * a64::X18_ISLAND_BYTES
            + tls_sites as u64 * a64::TLS_ISLAND_BYTES
            + undefined * (a64::ERRNO_THUNK_BYTES + 8)
            + 128,
    );
    let (base, island_start) = (BASE, BASE + image_end);
    let end = island_start + island_bytes;
    if island_start + island_bytes - base > (1 << 27) {
        return Err("shared object too large for direct branches to its islands".into());
    }

    if let Some(next) = image::next_mapping(base) {
        if next < end {
            return Err(format!(
                "address range {base:#x}..{end:#x} is not free; refusing to overwrite"
            ));
        }
    }
    let mut addr = base;
    let kr = unsafe {
        image::mach_vm_allocate(
            image::mach_task_self_,
            &mut addr,
            end - base,
            image::VM_FLAGS_FIXED,
        )
    };
    if kr != 0 || addr != base {
        return Err(format!(
            "mach_vm_allocate(fixed, no overwrite) for the shared object failed: kr={kr}"
        ));
    }

    unsafe {
        for l in &v.loads {
            std::ptr::copy_nonoverlapping(
                i.elf[l.off as usize..(l.off + l.filesz) as usize].as_ptr(),
                (base + l.va) as *mut u8,
                l.filesz as usize,
            );
        }
        let mut cursor = island_start;
        let trap_common = cursor;
        image::put(&mut cursor, &a64::trap_common(i.trap_handler));
        cursor = (cursor + 7) & !7;
        let tlsdesc_resolver = cursor;
        image::put(&mut cursor, &[0xf940_0400, a64::RET]); // ldr x0, [x0, #8]; ret

        // --- symbols ---------------------------------------------------------
        let mut exports = BTreeMap::new();
        let mut address_of = BTreeMap::<u64, u64>::new();
        let (mut bound, mut trapping, mut trap_slot) = (0usize, Vec::new(), 0usize);
        let mut data_traps = BTreeMap::new();
        for k in 1..nsyms {
            let (name, shndx, value, info) = v.symbol(k, nsyms)?;
            let (binding, kind) = (info >> 4, info & 15);
            if shndx != 0 {
                if kind == 6 {
                    continue; // TLS symbols are reached through TLSDESC only
                }
                if value >= link_hi {
                    return Err(format!("symbol {name} outside the image"));
                }
                address_of.insert(k, base + value);
                if binding != 0 && (kind == 1 || kind == 2) {
                    exports.insert(name, base + value);
                }
                continue;
            }
            let address = match (i.resolve)(&name) {
                Some(Binding::Raw(a)) => {
                    bound += 1;
                    a
                }
                Some(Binding::Thunk(target)) => {
                    bound += 1;
                    let at = cursor;
                    image::put(&mut cursor, &a64::errno_thunk(target, i.errno_sync));
                    at
                }
                Some(Binding::WeakNull) if binding == 2 => 0,
                Some(Binding::WeakNull) => return Err(format!("{name} is not weak")),
                None if binding == 2 && kind != 2 => 0,
                None if kind == 1 => {
                    // Unimplemented DATA import: an address in the unmapped
                    // first 4 GiB; any access faults and the reporter names it.
                    data_traps.insert(0xdeae_0000 + 16 * k, name.clone());
                    trapping.push(name);
                    trap_slot += 1;
                    0xdeae_0000 + 16 * k
                }
                None => {
                    let at = cursor;
                    let index = (i.trap_index_base + trap_slot) as u32;
                    image::put(&mut cursor, &a64::trap_stub(index, at, trap_common)?);
                    trapping.push(name);
                    trap_slot += 1;
                    at
                }
            };
            address_of.insert(k, address);
        }

        // --- relocations (exact; unknown types are refused) -------------------
        let mut counts = BTreeMap::<&'static str, u64>::new();
        for (table, size) in tables {
            if size % 24 != 0 || size > MAX_TABLE {
                return Err("malformed relocation table".into());
            }
            let off = v.offset(table, size)?;
            for k in 0..size / 24 {
                let r = off + 24 * k;
                let (target, info, addend) =
                    (u64_at(v.d, r)?, u64_at(v.d, r + 8)?, u64_at(v.d, r + 16)?);
                let (kind, sym) = (info & 0xffff_ffff, info >> 32);
                let width = if kind == 1031 { 16 } else { 8 };
                if target < data.va
                    || target
                        .checked_add(width)
                        .is_none_or(|e| e > data.va + data.memsz)
                {
                    return Err(format!(
                        "relocation target {target:#x} outside the writable segment"
                    ));
                }
                let symbol = |what: &str| -> Result<u64, String> {
                    address_of
                        .get(&sym)
                        .copied()
                        .ok_or_else(|| format!("{what} relocation against unusable symbol {sym}"))
                };
                let at = base + target;
                let name = match kind {
                    1027 => {
                        if sym != 0 || addend >= link_hi {
                            return Err("malformed RELATIVE relocation".into());
                        }
                        image::w64(at, base + addend);
                        "RELATIVE"
                    }
                    257 => {
                        image::w64(at, symbol("ABS64")?.wrapping_add(addend));
                        "ABS64"
                    }
                    1025 | 1026 => {
                        if addend != 0 {
                            return Err("GOT relocation with an addend".into());
                        }
                        image::w64(at, symbol("GOT")?);
                        if kind == 1025 {
                            "GLOB_DAT"
                        } else {
                            "JUMP_SLOT"
                        }
                    }
                    1031 => {
                        // Local-dynamic only: the object's own TLS block.
                        if sym != 0 || addend >= tls_bytes {
                            return Err("unsupported TLSDESC relocation".into());
                        }
                        image::w64(at, tlsdesc_resolver);
                        image::w64(at + 8, i.tls_offset + addend);
                        "TLSDESC"
                    }
                    other => return Err(format!("unsupported relocation type {other}")),
                };
                *counts.entry(name).or_insert(0) += 1;
            }
        }

        image::patch_code(
            m,
            text.va,
            text.va + text.filesz,
            base,
            &mut cursor,
            i.tp_slot,
            i.x18_slot,
            false,
        )?;
        if cursor > end {
            return Err("shared object island region overflow".into());
        }

        // --- protections ------------------------------------------------------
        let text_end = image::up(text.va + text.memsz);
        let data_start = image::down(data.va);
        let relro_end = match v.relro {
            Some((va, size)) if va == data.va && size <= data.memsz => image::down(va + size),
            Some(_) => return Err("PT_GNU_RELRO is not a prefix of the data segment".into()),
            None => data_start,
        };
        for (start, stop, prot) in [
            (0, text_end, image::PROT_RX),
            (text_end, data_start, image::PROT_NONE),
            (data_start, relro_end, image::PROT_R),
            (image_end, image_end + island_bytes, image::PROT_RX),
        ] {
            if stop > start
                && image::mprotect((base + start) as *mut c_void, (stop - start) as usize, prot)
                    != 0
            {
                return Err("mprotect failed for the shared object".into());
            }
            if prot == image::PROT_RX {
                image::sys_icache_invalidate(
                    (base + start) as *mut c_void,
                    (stop - start) as usize,
                );
            }
        }
        for (at, want) in [
            (base, image::PROT_RX),
            (island_start, image::PROT_RX),
            (base + relro_end, image::PROT_RW),
        ] {
            match image::region_protection(at) {
                Some((_, _, got)) if got == want => {}
                other => {
                    return Err(format!(
                        "shared object protection check failed at {at:#x}: {other:?}"
                    ))
                }
            }
        }

        // --- constructors: DT_INIT, then DT_INIT_ARRAY in order ---------------
        let mut init = Vec::new();
        let executable = |a: u64| a % 4 == 0 && a >= text.va && a + 4 <= text.va + text.filesz;
        if let Some(&f) = v.tags.get(&12) {
            if !executable(f) {
                return Err("DT_INIT outside the text segment".into());
            }
            init.push(base + f);
        }
        if let Some(&array) = v.tags.get(&25) {
            let size = v.tag(27)?;
            if size % 8 != 0
                || array < data.va
                || array
                    .checked_add(size)
                    .is_none_or(|e| e > data.va + data.filesz)
            {
                return Err("malformed DT_INIT_ARRAY".into());
            }
            for k in 0..size / 8 {
                let f = image::r64(base + array + 8 * k); // already relocated
                if f < base || !executable(f - base) {
                    return Err("constructor outside the text segment".into());
                }
                init.push(f);
            }
        }
        Ok(Dso {
            base,
            image_end: island_start,
            end,
            phdr: base + v.phoff,
            phnum: v.phnum,
            init,
            exports,
            tls_bytes,
            relocations: counts,
            imports_bound: bound,
            imports_trapping: trapping,
            data_traps,
            x18_sites,
            tls_sites,
        })
    }
}
