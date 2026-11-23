// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: map NVIDIA's Linux-aarch64 `tileiras` ET_EXEC into
// this process.
//
// The ELF is linked at 0x400000. Darwin arm64 refuses to exec any image whose
// __PAGEZERO is smaller than 4 GiB (measured on macOS 26.6: SIGKILL at exec for
// -pagezero_size 0x4000 .. 0x80000000) and a 4 GiB __PAGEZERO puts 0x400000
// below the task's minimum address (mach_vm_allocate -> KERN_INVALID_ADDRESS).
// The image is therefore loaded at `link address + SLIDE`:
//   * code is position independent already (adrp/adr/bl; the only absolute
//     literals are the three in glibc's `_start`, listed as abs64 rows);
//   * link-time pointers in data are rebased by the SHA-bound rule in the
//     manifest (`# rebase` lines), recomputed and compared here;
//   * a pointer that escaped the rule cannot be used silently: nothing is
//     mapped below 4 GiB, so dereferencing or calling it faults and the fault
//     reporter names it.
//
// The reservation is made with mach_vm_allocate(VM_FLAGS_FIXED) WITHOUT
// VM_FLAGS_OVERWRITE: it fails if any page of the range is already mapped.

use std::ffi::c_void;

use tileiras_native::a64::{self, X18Shape};
use tileiras_native::patches::{Manifest, PatchKind};
use tileiras_native::{sha256, LoadPlan};

/// Fixed, deterministic slide. Measured on macOS 26.6 arm64: 0x10_0000_0000 ..
/// 0x70_0000_0000 is a kernel PROT_NONE reservation in every process and the
/// dyld shared cache sits at 0x1_8000_0000 .. 0x3_0000_0000; 448 GiB is clear of
/// both, of the main image and of the malloc zones. If it is ever occupied the
/// reservation fails and the helper refuses to run (no fallback address).
pub const SLIDE: u64 = 0x70_0000_0000;
pub const PAGE: u64 = 16384;
pub(crate) const VM_FLAGS_FIXED: i32 = 0;
pub(crate) const PROT_NONE: i32 = 0;
pub(crate) const PROT_R: i32 = 1;
pub(crate) const PROT_RW: i32 = 3;
pub(crate) const PROT_RX: i32 = 5;

extern "C" {
    pub(crate) static mach_task_self_: u32;
    pub(crate) fn mach_vm_allocate(task: u32, addr: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_region(
        task: u32,
        addr: *mut u64,
        size: *mut u64,
        flavor: i32,
        info: *mut i32,
        count: *mut u32,
        object: *mut u32,
    ) -> i32;
    pub(crate) fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    pub(crate) fn sys_icache_invalidate(start: *mut c_void, len: usize);
}

/// How one import was bound.
pub enum Binding {
    /// Darwin function called through an errno-translating thunk.
    Thunk(u64),
    /// Address used as is (functions that read stack arguments, never return,
    /// or return twice; data cells).
    Raw(u64),
    /// Weak symbol left undefined (address 0), as ld.so would.
    WeakNull,
}

pub struct Region {
    pub name: &'static str,
    pub start: u64,
    pub end: u64,
    pub want: i32,
}

pub struct Loaded {
    pub slide: u64,
    pub reserve_start: u64,
    pub reserve_end: u64,
    pub entry: u64,
    pub regions: Vec<Region>,
    pub rebased: u64,
    /// In-range words left unslid on purpose (`norebase` rows).
    pub not_rebased: u64,
    /// Declared RELRO bytes that stay writable because they share a 16 KiB host
    /// page with ordinary data (link-time addresses).
    pub relro_writable_tail: (u64, u64),
    pub x18_sites: usize,
    pub tls_sites: usize,
    pub imports_bound: Vec<String>,
    pub imports_trapping: Vec<String>,
    pub imports_weak_null: Vec<String>,
    pub import_names: Vec<String>,
}

pub(crate) fn down(v: u64) -> u64 {
    v & !(PAGE - 1)
}
pub(crate) fn up(v: u64) -> u64 {
    (v + PAGE - 1) & !(PAGE - 1)
}

/// Protection of the VM region containing `addr`, or None if unmapped.
pub fn region_protection(addr: u64) -> Option<(u64, u64, i32)> {
    let (mut start, mut size, mut info, mut count, mut object) =
        (addr, 0u64, [0i32; 16], 9u32, 0u32);
    let kr = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut start,
            &mut size,
            9, // VM_REGION_BASIC_INFO_64
            info.as_mut_ptr(),
            &mut count,
            &mut object,
        )
    };
    (kr == 0 && start <= addr).then_some((start, start + size, info[0]))
}

/// First mapped address at or above `addr` (None: nothing above).
pub(crate) fn next_mapping(addr: u64) -> Option<u64> {
    let (mut start, mut size, mut info, mut count, mut object) =
        (addr, 0u64, [0i32; 16], 9u32, 0u32);
    let kr = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut start,
            &mut size,
            9,
            info.as_mut_ptr(),
            &mut count,
            &mut object,
        )
    };
    (kr == 0).then_some(start)
}

pub(crate) unsafe fn w32(addr: u64, value: u32) {
    (addr as *mut u32).write_unaligned(value)
}
pub(crate) unsafe fn r32(addr: u64) -> u32 {
    (addr as *const u32).read_unaligned()
}
pub(crate) unsafe fn w64(addr: u64, value: u64) {
    (addr as *mut u64).write_unaligned(value)
}
pub(crate) unsafe fn r64(addr: u64) -> u64 {
    (addr as *const u64).read_unaligned()
}
pub(crate) unsafe fn put(at: &mut u64, words: &[u32]) {
    for w in words {
        w32(*at, *w);
        *at += 4;
    }
}

pub struct Inputs<'a> {
    pub elf: &'a [u8],
    pub plan: &'a LoadPlan,
    pub manifest: &'a Manifest,
    /// TSD byte offsets (pthread key * 8) of the Linux thread pointer and of
    /// virtual x18.
    pub tp_slot: u64,
    pub x18_slot: u64,
    pub resolve: &'a dyn Fn(&str) -> Option<Binding>,
    pub trap_handler: u64,
    pub errno_sync: u64,
}

/// Reserve, populate, relocate, patch and protect the image. Executes nothing.
pub fn load(i: &Inputs) -> Result<Loaded, String> {
    let (plan, m) = (i.plan, i.manifest);
    if plan.host_page_size != PAGE {
        return Err("plan was not made for 16 KiB pages".into());
    }
    if m.elf_sha256 != plan.input_sha256 || sha256::hex(i.elf) != plan.input_sha256 {
        return Err("patch manifest / ELF / plan SHA-256 mismatch".into());
    }
    let logical_hi = plan
        .segments
        .iter()
        .map(|s| s.virtual_address + s.memory_size)
        .max()
        .ok_or("no segments")?;
    if m.image_lo != plan.load_base || m.image_hi != logical_hi {
        return Err("patch manifest image bounds differ from the plan".into());
    }
    let (text, data) = match &plan.segments[..] {
        [t, d] if t.execute && !t.write && d.write && !d.execute && t.page_end <= d.page_start => {
            (t, d)
        }
        _ => return Err("expected one R-X and one RW- load segment on disjoint pages".into()),
    };
    let relro = match &plan.relro[..] {
        [r] if r.virtual_address == data.virtual_address => r,
        _ => return Err("expected one PT_GNU_RELRO at the start of the data segment".into()),
    };
    let x18_sites = m.count(|k| matches!(k, PatchKind::X18(_)));
    let tls_sites = m.count(|k| *k == PatchKind::Tls);
    let island_bytes = up(x18_sites as u64 * a64::X18_ISLAND_BYTES
        + tls_sites as u64 * a64::TLS_ISLAND_BYTES
        + plan.imports.len() as u64 * (a64::ERRNO_THUNK_BYTES + 8)
        + 64);
    let slide = SLIDE;
    let start = plan.load_base + slide;
    let island_start = plan.load_end + slide;
    let end = island_start + island_bytes;

    // --- non-destructive reservation ---------------------------------------
    if let Some(next) = next_mapping(start) {
        if next < end {
            return Err(format!(
                "address range {start:#x}..{end:#x} is not free (mapping at {next:#x}); refusing to overwrite"
            ));
        }
    }
    let mut addr = start;
    let kr = unsafe { mach_vm_allocate(mach_task_self_, &mut addr, end - start, VM_FLAGS_FIXED) };
    if kr != 0 || addr != start {
        return Err(format!(
            "mach_vm_allocate(fixed, no overwrite) of {start:#x}..{end:#x} failed: kr={kr}"
        ));
    }

    unsafe {
        // --- segment bytes (anonymous memory is zero-filled: .bss, padding) --
        for s in &plan.segments {
            let src = &i.elf[s.file_offset as usize..(s.file_offset + s.file_size) as usize];
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (s.virtual_address + slide) as *mut u8,
                src.len(),
            );
        }

        // --- data pointers: recompute the manifest's rule and compare --------
        // `norebase` rows: aligned in-range words whose value is contradicted
        // as an address (see gen_patch_manifest.py); they keep their bytes.
        let keep: std::collections::BTreeMap<u64, u64> = m
            .rows
            .iter()
            .filter(|r| r.kind == PatchKind::NoRebase)
            .map(|r| (r.address, (r.new_word as u64) << 32 | r.old_word as u64))
            .collect();
        let (mut rebased, mut kept) = (0u64, 0u64);
        for range in &m.rebase {
            let skip = range.address.wrapping_neg() % 8;
            if range.address < data.virtual_address
                || range.size < skip
                || range
                    .address
                    .checked_add(range.size)
                    .is_none_or(|e| e > data.virtual_address + data.file_size)
            {
                return Err(format!(
                    "rebase range {} leaves the data segment",
                    range.section
                ));
            }
            let (mut count, mut digest_input) = (0u64, Vec::new());
            for k in 0..(range.size - skip) / 8 {
                let va = range.address + skip + 8 * k;
                let v = r64(va + slide);
                if (m.image_lo..m.image_hi).contains(&v) {
                    if let Some(expected) = keep.get(&va) {
                        if *expected != v {
                            return Err(format!("norebase row {va:#x}: image word differs"));
                        }
                        kept += 1;
                        continue;
                    }
                    w64(va + slide, v + slide);
                    count += 1;
                    digest_input.extend_from_slice(&va.to_le_bytes());
                }
            }
            if count != range.count || sha256::hex(&digest_input) != range.sha256 {
                return Err(format!(
                    "rebase set of {} differs from the manifest ({count} words)",
                    range.section
                ));
            }
            rebased += count;
        }
        if kept != keep.len() as u64 {
            return Err("a norebase row names a word outside the rebase rule".into());
        }

        // --- islands, thunks, trap stubs ------------------------------------
        let mut cursor = island_start;
        let trap_common = cursor;
        put(&mut cursor, &a64::trap_common(i.trap_handler));
        cursor = (cursor + 7) & !7;

        // --- imports ---------------------------------------------------------
        let import_names: Vec<String> = plan.imports.iter().map(|s| s.name.clone()).collect();
        let (mut bound, mut trapping, mut weak_null) = (Vec::new(), Vec::new(), Vec::new());
        let mut address_of = std::collections::BTreeMap::<u32, u64>::new();
        for (index, import) in plan.imports.iter().enumerate() {
            let value = match (i.resolve)(&import.name) {
                Some(Binding::Raw(a)) => {
                    bound.push(import.name.clone());
                    a
                }
                Some(Binding::Thunk(target)) => {
                    bound.push(import.name.clone());
                    let at = cursor;
                    put(&mut cursor, &a64::errno_thunk(target, i.errno_sync));
                    at
                }
                Some(Binding::WeakNull) if import.weak => {
                    weak_null.push(import.name.clone());
                    0
                }
                Some(Binding::WeakNull) => return Err(format!("{} is not weak", import.name)),
                None if import.symbol_type == 1 => {
                    // Unimplemented DATA import: an address in the unmapped
                    // first 4 GiB; any access faults and the reporter names it.
                    trapping.push(import.name.clone());
                    0xdead_0000 + 16 * index as u64
                }
                None => {
                    trapping.push(import.name.clone());
                    let at = cursor;
                    put(&mut cursor, &a64::trap_stub(index as u32, at, trap_common)?);
                    at
                }
            };
            address_of.insert(import.symbol_index, value);
        }
        for r in &plan.relocations {
            if ![1025, 1026].contains(&r.kind) || r.addend != 0 {
                return Err(format!(
                    "unsupported relocation {} for {}",
                    r.kind, r.symbol_name
                ));
            }
            if r.target < data.virtual_address
                || r.target + 8 > data.virtual_address + data.file_size
            {
                return Err("import relocation outside the data segment".into());
            }
            let value = *address_of
                .get(&r.symbol_index)
                .ok_or_else(|| format!("relocation against unplanned symbol {}", r.symbol_name))?;
            w64(r.target + slide, value);
        }

        // --- code patches (old word checked first: fail closed) --------------
        patch_code(
            m,
            text.virtual_address,
            text.virtual_address + text.file_size,
            slide,
            &mut cursor,
            i.tp_slot,
            i.x18_slot,
            true,
        )?;
        if cursor > end {
            return Err("island region overflow".into());
        }

        // --- protections (W^X: nothing is ever writable and executable) ------
        let relro_end = down(relro.virtual_address + relro.size);
        let regions = vec![
            Region {
                name: "text",
                start: text.page_start + slide,
                end: text.page_end + slide,
                want: PROT_RX,
            },
            Region {
                name: "guard",
                start: text.page_end + slide,
                end: data.page_start + slide,
                want: PROT_NONE,
            },
            Region {
                name: "relro",
                start: data.page_start + slide,
                end: relro_end + slide,
                want: PROT_R,
            },
            Region {
                name: "data",
                start: relro_end + slide,
                end: data.page_end + slide,
                want: PROT_RW,
            },
            Region {
                name: "islands",
                start: island_start,
                end,
                want: PROT_RX,
            },
        ];
        if regions.last().map(|r| r.start) != Some(data.page_end + slide) {
            return Err("island region is not adjacent to the image".into());
        }
        for r in &regions {
            if r.want != PROT_RW && r.end > r.start {
                if mprotect(r.start as *mut c_void, (r.end - r.start) as usize, r.want) != 0 {
                    return Err(format!(
                        "mprotect({}, {:#x}) failed (hardened runtime would need MAP_JIT)",
                        r.name, r.want
                    ));
                }
            }
            if r.want == PROT_RX {
                sys_icache_invalidate(r.start as *mut c_void, (r.end - r.start) as usize);
            }
        }
        Ok(Loaded {
            slide,
            reserve_start: start,
            reserve_end: end,
            entry: plan.entry + slide,
            regions,
            rebased,
            not_rebased: kept,
            relro_writable_tail: (relro_end, relro.virtual_address + relro.size),
            x18_sites,
            tls_sites,
            imports_bound: bound,
            imports_trapping: trapping,
            imports_weak_null: weak_null,
            import_names,
        })
    }
}

/// Apply the manifest's code rows to a mapped, still writable image.
/// `text_lo..text_hi` are link-time bounds of the file-backed text segment.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn patch_code(
    m: &Manifest,
    text_lo: u64,
    text_hi: u64,
    slide: u64,
    cursor: &mut u64,
    tp_slot: u64,
    x18_slot: u64,
    allow_abs64: bool,
) -> Result<(), String> {
    for row in &m.rows {
        if row.kind == PatchKind::NoRebase {
            if !allow_abs64 {
                return Err("a shared object manifest cannot carry norebase rows".into());
            }
            continue; // data word, handled by the rebase pass
        }
        if row.address < text_lo || row.address + 8 > text_hi {
            return Err(format!(
                "patch row {:#x} is outside the text segment",
                row.address
            ));
        }
        let site = row.address + slide;
        if r32(site) != row.old_word {
            return Err(format!("patch row {:#x}: image word differs", row.address));
        }
        match row.kind {
            PatchKind::Abs64 => {
                let v = r64(site);
                if !allow_abs64
                    || v >> 32 != row.new_word as u64
                    || !(m.image_lo..m.image_hi).contains(&v)
                {
                    return Err(format!("abs64 row {:#x}: not admissible", row.address));
                }
                w64(site, v + slide);
            }
            PatchKind::NoRebase => unreachable!("filtered above"),
            PatchKind::Tls => {
                let island = *cursor;
                put(cursor, &a64::tls_island(row.r, tp_slot, site, island)?);
                w32(site, a64::b(site, island)?);
            }
            PatchKind::X18(shape) => {
                let island = *cursor;
                let words =
                    a64::x18_island(shape, row.new_word, row.r, row.r2, x18_slot, site, island)?;
                put(cursor, &words);
                *cursor = island + a64::X18_ISLAND_BYTES;
                w32(
                    site,
                    if shape == X18Shape::Blr {
                        a64::bl(site, island)?
                    } else {
                        a64::b(site, island)?
                    },
                );
            }
        }
    }
    Ok(())
}

/// Re-read every region's protection from the kernel.
pub fn verify(loaded: &Loaded) -> Result<Vec<(String, u64, u64, i32, i32)>, String> {
    let mut out = Vec::new();
    for r in &loaded.regions {
        if r.end == r.start {
            continue;
        }
        let mut at = r.start;
        let mut seen = r.want;
        while at < r.end {
            let (_, region_end, prot) =
                region_protection(at).ok_or_else(|| format!("{} is not mapped", r.name))?;
            if prot != r.want {
                return Err(format!(
                    "{} at {at:#x}: kernel reports protection {prot}, wanted {}",
                    r.name, r.want
                ));
            }
            if prot & 2 != 0 && prot & 4 != 0 {
                return Err("W+X page".into());
            }
            seen = prot;
            at = region_end;
        }
        out.push((r.name.to_string(), r.start, r.end, r.want, seen));
    }
    Ok(out)
}
