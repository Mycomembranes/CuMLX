/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level synchronous tensor-core MMA (`mma.sync`), sm_80+.
//!
//! This is the tensor-core path of consumer Blackwell (sm_120, e.g. RTX 50xx), which has
//! `mma.sync` but neither `wgmma` (sm_90a) nor `tcgen05` (sm_100a). Each call is ONE warp
//! instruction: all 32 lanes must execute it together (it is lowered as convergent inline
//! PTX), with the operands in the fragment layout below.
//!
//! # `mma_m16n8k16_f32_bf16`
//!
//! `D[16x8] = A[16x16] · B[16x8] + C[16x8]`, A/B bf16, C/D f32.
//!
//! ```text
//! PTX: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//!          {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
//!
//! lane = 0..31, g = lane >> 2 (0..7), t = lane & 3 (0..3).
//! A regs are packed bf16 pairs (low half = smaller k index):
//!   a0 = A[g   ][2t, 2t+1]     a1 = A[g+8][2t, 2t+1]
//!   a2 = A[g   ][2t+8, 2t+9]   a3 = A[g+8][2t+8, 2t+9]
//! B regs ("col" = B stored k-contiguous per column n):
//!   b0 = B[2t, 2t+1][g]        b1 = B[2t+8, 2t+9][g]
//! C / D (f32):
//!   c0 = C[g][2t]  c1 = C[g][2t+1]  c2 = C[g+8][2t]  c3 = C[g+8][2t+1]
//! ```
//!
//! Pack bf16 pairs with [`crate::tcgen05::cvt_f32x2_bf16x2`] (`cvt.rn.bf16x2.f32`,
//! round-to-nearest-even; first argument -> low half).

use crate::cusimd::CuSimd;

/// The four f32 accumulator registers of one `m16n8k16` fragment (`d0..d3`).
pub type MmaF32x4 = CuSimd<f32, 4>;

/// Warp-synchronous `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
///
/// Returns `D = A·B + C` for this lane's fragment (see the module docs for the layout).
/// `a0..a3` / `b0..b1` are packed bf16 pairs, `c0..c3` the f32 accumulators.
///
/// Must be executed by all 32 lanes of the warp, converged (no divergent branch around it).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn mma_m16n8k16_f32_bf16(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
) -> MmaF32x4 {
    let _ = (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    unreachable!("mma_m16n8k16_f32_bf16 called outside CUDA kernel context")
}
