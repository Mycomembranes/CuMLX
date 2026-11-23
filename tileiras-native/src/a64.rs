// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: the few AArch64 encoders the tileiras loader needs
// for its islands, import thunks and trap stubs. Every constant below was
// checked against `llvm-mc -show-encoding` (llvm@21); the unit tests pin them.

/// Per-island byte budget for an x18 site (the largest shape is 12 words).
pub const X18_ISLAND_BYTES: u64 = 48;
/// `mrs xN, tpidrro_el0; and; ldr; b`.
pub const TLS_ISLAND_BYTES: u64 = 16;
pub const NOP: u32 = 0xd503_201f;
pub const RET: u32 = 0xd65f_03c0;

fn range(delta: i64, bits: u32) -> Result<u32, String> {
    if delta & 3 != 0 || !(-(1i64 << (bits + 1))..(1i64 << (bits + 1))).contains(&delta) {
        return Err(format!("branch displacement {delta:#x} out of range"));
    }
    Ok(((delta >> 2) as u32) & ((1u32 << bits) - 1))
}
pub fn b(from: u64, to: u64) -> Result<u32, String> {
    Ok(0x1400_0000 | range(to as i64 - from as i64, 26)?)
}
pub fn bl(from: u64, to: u64) -> Result<u32, String> {
    Ok(0x9400_0000 | range(to as i64 - from as i64, 26)?)
}
pub fn br(rn: u32) -> u32 {
    0xd61f_0000 | (rn << 5)
}
pub fn blr(rn: u32) -> u32 {
    0xd63f_0000 | (rn << 5)
}
/// `stur xT, [sp, #simm9]` / `ldur xT, [sp, #simm9]` (Darwin red zone slots).
pub fn stur_sp(rt: u32, simm9: i32) -> u32 {
    0xf800_0000 | (((simm9 as u32) & 0x1ff) << 12) | (31 << 5) | rt
}
pub fn ldur_sp(rt: u32, simm9: i32) -> u32 {
    0xf840_0000 | (((simm9 as u32) & 0x1ff) << 12) | (31 << 5) | rt
}
pub fn mrs_tpidrro(rt: u32) -> u32 {
    0xd53b_d060 | rt
}
/// `mrs xT, tpidr_el0`, the instruction being replaced.
pub fn mrs_tpidr(rt: u32) -> u32 {
    0xd53b_d040 | rt
}
/// `and xD, xN, #0xfffffffffffffff8` (Darwin keeps the cpu number in bits 0..2).
pub fn and_not7(rd: u32, rn: u32) -> u32 {
    0x927d_f000 | (rn << 5) | rd
}
pub fn ldr_off(rt: u32, rn: u32, byte_offset: u64) -> Result<u32, String> {
    if byte_offset % 8 != 0 || byte_offset / 8 > 4095 {
        return Err(format!("ldr offset {byte_offset:#x} not encodable"));
    }
    Ok(0xf940_0000 | ((byte_offset / 8) as u32) << 10 | (rn << 5) | rt)
}
pub fn str_off(rt: u32, rn: u32, byte_offset: u64) -> Result<u32, String> {
    Ok(ldr_off(rt, rn, byte_offset)? & !0x0040_0000)
}
/// `ldr xT, <pc + delta>` (64-bit literal).
pub fn ldr_literal(rt: u32, delta: i64) -> Result<u32, String> {
    Ok(0x5800_0000 | (range(delta, 19)? << 5) | rt)
}
pub fn movz_w(rd: u32, imm16: u32) -> u32 {
    0x5280_0000 | ((imm16 & 0xffff) << 5) | rd
}
pub fn mov_x(rd: u32, rm: u32) -> u32 {
    0xaa00_03e0 | (rm << 16) | rd
}
fn sext(value: u64, bits: u32) -> i64 {
    ((value << (64 - bits)) as i64) >> (64 - bits)
}
pub fn adrp_page(word: u32, pc: u64) -> u64 {
    let imm = sext(
        ((((word >> 5) & 0x7ffff) as u64) << 2) | ((word >> 29) & 3) as u64,
        21,
    );
    ((pc & !0xfff) as i64 + (imm << 12)) as u64
}
pub fn adrp_retarget(word: u32, pc: u64, page: u64) -> Result<u32, String> {
    let delta = (page as i64 - (pc & !0xfff) as i64) >> 12;
    if !(-(1 << 20)..(1 << 20)).contains(&delta) {
        return Err("adrp out of range".into());
    }
    let imm = (delta as u32) & 0x1f_ffff;
    Ok((word & !((3 << 29) | (0x7ffff << 5))) | ((imm & 3) << 29) | ((imm >> 2) << 5))
}
/// Branch target of cbz/cbnz (imm19) at `pc`.
pub fn cb_target(word: u32, pc: u64) -> u64 {
    (pc as i64 + sext(((word >> 5) & 0x7ffff) as u64, 19) * 4) as u64
}
/// Branch target of tbz/tbnz (imm14) at `pc`.
pub fn tb_target(word: u32, pc: u64) -> u64 {
    (pc as i64 + sext(((word >> 5) & 0x3fff) as u64, 14) * 4) as u64
}
pub fn cb_with_delta(word: u32, delta: i64) -> Result<u32, String> {
    Ok((word & !(0x7ffff << 5)) | (range(delta, 19)? << 5))
}
pub fn tb_with_delta(word: u32, delta: i64) -> Result<u32, String> {
    Ok((word & !(0x3fff << 5)) | (range(delta, 14)? << 5))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X18Shape {
    Plain,
    Adrp,
    Cb,
    Tb,
    Blr,
}

/// Island for one x18 instruction. "Virtual x18" lives in the calling thread's
/// TSD slot `slot_bytes / 8`; R replaces x18 in the instruction and R2 holds the
/// TSD base. Both are parked in the 128-byte Darwin red zone below sp for the
/// duration of the island only (Linux code never addresses below sp, and the
/// generator refuses instructions that write sp). NZCV is never touched.
///
/// `new_word` is the instruction with x18 already replaced by R; `site` and
/// `island` are run-time addresses. For `Blr` the site is rewritten to
/// `bl island`, so x30 already holds the return address and x16/x17 (dead at
/// any call by AAPCS64) carry the target.
pub fn x18_island(
    shape: X18Shape,
    new_word: u32,
    r: u32,
    r2: u32,
    slot_bytes: u64,
    site: u64,
    island: u64,
) -> Result<Vec<u32>, String> {
    if r == r2 || [r, r2].iter().any(|x| !(9..=15).contains(x)) {
        return Err("illegal scratch registers".into());
    }
    if shape == X18Shape::Blr {
        return Ok(vec![
            mrs_tpidrro(17),
            and_not7(17, 17),
            ldr_off(16, 17, slot_bytes)?,
            br(16),
        ]);
    }
    let mut w = vec![
        stur_sp(r, -16),
        stur_sp(r2, -24),
        mrs_tpidrro(r2),
        and_not7(r2, r2),
        ldr_off(r, r2, slot_bytes)?,
    ];
    let at = |w: &Vec<u32>| island + 4 * w.len() as u64;
    match shape {
        X18Shape::Plain | X18Shape::Adrp => {
            if shape == X18Shape::Adrp {
                let page = adrp_page(new_word, site);
                w.push(adrp_retarget(new_word, at(&w), page)?);
            } else {
                w.push(new_word);
            }
            w.push(str_off(r, r2, slot_bytes)?);
            w.push(ldur_sp(r2, -24));
            w.push(ldur_sp(r, -16));
            w.push(b(at(&w), site + 4)?);
        }
        X18Shape::Cb | X18Shape::Tb => {
            // [5] cb/tb R, taken ; [6..8] restore, b fallthrough ; [9..11] restore, b target
            let (target, branch) = if shape == X18Shape::Cb {
                (cb_target(new_word, site), cb_with_delta(new_word, 16)?)
            } else {
                (tb_target(new_word, site), tb_with_delta(new_word, 16)?)
            };
            w.push(branch);
            w.push(ldur_sp(r2, -24));
            w.push(ldur_sp(r, -16));
            w.push(b(at(&w), site + 4)?);
            w.push(ldur_sp(r2, -24));
            w.push(ldur_sp(r, -16));
            w.push(b(at(&w), target)?);
        }
        X18Shape::Blr => unreachable!(),
    }
    debug_assert!(4 * w.len() as u64 <= X18_ISLAND_BYTES);
    Ok(w)
}

/// `mrs xN, tpidr_el0` replacement: load the Linux thread pointer from TSD.
pub fn tls_island(n: u32, slot_bytes: u64, site: u64, island: u64) -> Result<Vec<u32>, String> {
    if n == 18 || n > 30 {
        return Err("illegal TLS destination register".into());
    }
    Ok(vec![
        mrs_tpidrro(n),
        and_not7(n, n),
        ldr_off(n, n, slot_bytes)?,
        b(island + 12, site + 4)?,
    ])
}

/// Call `target`, then `sync` (which moves a Darwin errno into the Linux
/// errno cell), preserving every result register (x0, x1, d0..d3). Not usable
/// for callees that read stack-passed arguments (the frame shifts them).
pub fn errno_thunk(target: u64, sync: u64) -> Vec<u32> {
    let mut w = vec![
        0xa9bf_7bfd,                  // stp x29, x30, [sp, #-16]!
        0x9100_03fd,                  // mov x29, sp
        ldr_literal(16, 48).unwrap(), // ldr x16, target
        blr(16),
        0xa9bd_07e0,                  // stp x0, x1, [sp, #-48]!
        0x6d01_07e0,                  // stp d0, d1, [sp, #16]
        0x6d02_0fe2,                  // stp d2, d3, [sp, #32]
        ldr_literal(16, 36).unwrap(), // ldr x16, sync
        blr(16),
        0x6d41_07e0, // ldp d0, d1, [sp, #16]
        0x6d42_0fe2, // ldp d2, d3, [sp, #32]
        0xa8c3_07e0, // ldp x0, x1, [sp], #48
        0xa8c1_7bfd, // ldp x29, x30, [sp], #16
        RET,
    ];
    debug_assert_eq!(w.len(), 14);
    w.push(target as u32);
    w.push((target >> 32) as u32);
    w.push(sync as u32);
    w.push((sync >> 32) as u32);
    w
}
pub const ERRNO_THUNK_BYTES: u64 = 72;

/// `movz w16, #index; b common`.
pub fn trap_stub(index: u32, at: u64, common: u64) -> Result<[u32; 2], String> {
    if index > 0xffff {
        return Err("too many imports".into());
    }
    Ok([movz_w(16, index), b(at + 4, common)?])
}
/// `mov x0, x16; ldr x17, handler; br x17; .quad handler`.
pub fn trap_common(handler: u64) -> [u32; 5] {
    [
        mov_x(0, 16),
        ldr_literal(17, 8).unwrap(),
        br(17),
        handler as u32,
        (handler >> 32) as u32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_match_llvm_mc() {
        assert_eq!(and_not7(8, 8), 0x927d_f108);
        assert_eq!(mrs_tpidrro(8), 0xd53b_d068);
        assert_eq!(ldr_off(9, 10, 2048).unwrap(), 0xf944_0149);
        assert_eq!(str_off(9, 10, 2048).unwrap(), 0xf904_0149);
        assert_eq!(stur_sp(9, -24), 0xf81e_83e9);
        assert_eq!(ldur_sp(9, -24), 0xf85e_83e9);
        assert_eq!(br(16), 0xd61f_0200);
        assert_eq!(blr(16), 0xd63f_0200);
        assert_eq!(mov_x(0, 16), 0xaa10_03e0);
        assert_eq!(ldr_literal(17, 8).unwrap(), 0x5800_0051);
        assert_eq!(ldr_literal(16, 40).unwrap(), 0x5800_0150);
        assert_eq!(movz_w(16, 5), 0x5280_00b0);
        assert_eq!(b(0x1000, 0x1008).unwrap(), 0x1400_0002);
        assert_eq!(bl(0x1008, 0x1000).unwrap(), 0x97ff_fffe);
        assert!(b(0, 1 << 28).is_err());
        assert!(ldr_off(0, 0, 4).is_err());
    }

    #[test]
    fn errno_thunk_literals_are_where_the_loads_point() {
        let t = errno_thunk(0x1111_2222_3333_4444, 0x5555_6666_7777_8888);
        assert_eq!(4 * t.len() as u64, ERRNO_THUNK_BYTES);
        // Decode both literal loads and check they land on the two quads.
        let lit = |i: usize| i as i64 * 4 + (((t[i] >> 5) & 0x7ffff) as i64) * 4;
        assert_eq!(lit(2), 14 * 4);
        assert_eq!(lit(7), 16 * 4);
        assert_eq!((t[14], t[15]), (0x3333_4444, 0x1111_2222));
        assert_eq!((t[16], t[17]), (0x7777_8888, 0x5555_6666));
    }

    #[test]
    fn plain_island_restores_scratch_and_returns_after_the_site() {
        // add x18, sp, #0x2c8 -> add x9, sp, #0x2c8
        let w = x18_island(X18Shape::Plain, 0x910b_23e9, 9, 10, 2048, 0x1000, 0x9000).unwrap();
        assert_eq!(w.len(), 10);
        assert_eq!(w[0], stur_sp(9, -16));
        assert_eq!(w[5], 0x910b_23e9);
        assert_eq!(w[6], str_off(9, 10, 2048).unwrap());
        assert_eq!(w[9], b(0x9024, 0x1004).unwrap());
    }

    #[test]
    fn conditional_island_keeps_both_edges() {
        // cbz x9, +0x40
        let word = 0xb400_0009 | (0x10 << 5);
        let w = x18_island(X18Shape::Cb, word, 9, 10, 2048, 0x2000, 0x9000).unwrap();
        assert_eq!(w.len(), 12);
        assert_eq!(cb_target(w[5], 0x9014), 0x9024);
        assert_eq!(w[8], b(0x9020, 0x2004).unwrap());
        assert_eq!(w[11], b(0x902c, 0x2040).unwrap());
        // tbnz w9, #3, -8
        let word = 0x3718_0009 | ((0x3fff - 1) << 5);
        assert_eq!(tb_target(word, 0x2000), 0x1ff8);
        let w = x18_island(X18Shape::Tb, word, 9, 10, 2048, 0x2000, 0x9000).unwrap();
        assert_eq!(tb_target(w[5], 0x9014), 0x9024);
        assert_eq!(w[11], b(0x902c, 0x1ff8).unwrap());
    }

    #[test]
    fn adrp_island_keeps_the_page() {
        let word = 0x9000_0009 | (3 << 29); // adrp x9, +3 pages
        let w = x18_island(X18Shape::Adrp, word, 9, 10, 2048, 0x10_0000, 0x500_0000).unwrap();
        assert_eq!(adrp_page(w[5], 0x500_0014), 0x10_3000);
    }

    #[test]
    fn tls_and_blr_islands() {
        let w = tls_island(8, 2056, 0x1000, 0x9000).unwrap();
        assert_eq!(w[0], 0xd53b_d068);
        assert_eq!(w[1], 0x927d_f108);
        assert_eq!(w[3], b(0x900c, 0x1004).unwrap());
        assert!(tls_island(18, 0, 0, 0).is_err());
        let w = x18_island(X18Shape::Blr, 0, 9, 10, 2048, 0, 0).unwrap();
        assert_eq!(w[3], br(16));
        assert!(x18_island(X18Shape::Plain, 0, 9, 9, 0, 0, 0).is_err());
        assert!(x18_island(X18Shape::Plain, 0, 16, 9, 0, 0, 0).is_err());
    }
}
