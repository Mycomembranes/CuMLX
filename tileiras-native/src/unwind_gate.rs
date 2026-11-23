// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for macOS ARM64: decide what a failed unwind-info lookup means.
//
// The guest's statically linked libgcc unwinder finds FDEs through
// `dl_iterate_phdr`, passing a `struct unw_eh_callback_data` whose first word
// is the pc it is looking for (return address - 1). Guest frames have ELF CFI;
// the helper's own frames do not: import thunks, x18 / TLS islands, and native
// libSystem / helper functions that called back into the guest (qsort
// comparator, pthread_once initialiser, key destructor, libNVVM constructor
// run by the dlopen shim). When a C++ exception has to cross one of those the
// unwinder gets "no FDE", reports end-of-stack, and the guest ends in
// std::terminate without any hint that the frame was the port's, not the
// program's. This classifier lets the dl_iterate_phdr shim turn exactly that
// case into a named diagnostic (exit 72) instead.

/// What the pc of a failed FDE lookup belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnwindPc {
    /// Guest code (tileiras or libNVVM image): a normal miss (PLT stub,
    /// `_start`), the unwinder's own business.
    Guest,
    /// The helper's guest-entry trampoline (below `main`, a thread start
    /// routine, an exit handler): the logical end of the guest stack, exactly
    /// like glibc's `__libc_start_main` / `start_thread` on Linux.
    StackRoot,
    /// An import thunk or an x18 / TLS island of the helper: no CFI exists.
    HelperIsland,
    /// Native executable code (libSystem or the helper itself) between two
    /// guest frames: no ELF CFI exists.
    NativeFrame,
    /// Not executable memory: the caller of dl_iterate_phdr was not the
    /// unwinder (its first data word is not a pc). Ignored.
    NotCode,
}

pub struct Layout<'a> {
    /// Guest images, `[start, end)`.
    pub guest: &'a [(u64, u64)],
    /// Helper island / thunk regions, `[start, end)`.
    pub islands: &'a [(u64, u64)],
    /// The guest-entry trampoline, `[start, end)`.
    pub stack_root: (u64, u64),
}

pub fn classify(pc: u64, layout: &Layout, is_executable: impl Fn(u64) -> bool) -> UnwindPc {
    let inside = |r: &(u64, u64)| r.0 <= pc && pc < r.1;
    if layout.islands.iter().any(inside) {
        UnwindPc::HelperIsland
    } else if layout.guest.iter().any(inside) {
        UnwindPc::Guest
    } else if inside(&layout.stack_root) {
        UnwindPc::StackRoot
    } else if is_executable(pc) {
        UnwindPc::NativeFrame
    } else {
        UnwindPc::NotCode
    }
}

impl UnwindPc {
    /// True when the lookup failure means "an exception cannot cross the port".
    pub fn is_fatal(self) -> bool {
        matches!(self, UnwindPc::HelperIsland | UnwindPc::NativeFrame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_port_frames_are_fatal() {
        let layout = Layout {
            guest: &[
                (0x70_0040_0000, 0x70_05b1_c000),
                (0x70_1000_0000, 0x70_1300_0000),
            ],
            islands: &[
                (0x70_05b1_c000, 0x70_05c0_0000),
                (0x70_1300_0000, 0x70_1310_0000),
            ],
            stack_root: (0x1_0000_1000, 0x1_0000_1014),
        };
        let exec = |pc: u64| (0x1_0000_0000..0x1_0010_0000).contains(&pc) || pc >= 0x70_0000_0000;
        let cases = [
            (0x70_0040_5dc0, UnwindPc::Guest),
            (0x70_1000_1000, UnwindPc::Guest),
            (0x70_05b1_c010, UnwindPc::HelperIsland),
            (0x70_1300_0000, UnwindPc::HelperIsland),
            (0x1_0000_1010, UnwindPc::StackRoot),
            (0x1_0000_2000, UnwindPc::NativeFrame),
            (0x16f_dff_000, UnwindPc::NotCode), // a stack address
            (0, UnwindPc::NotCode),
        ];
        for (pc, want) in cases {
            let got = classify(pc, &layout, exec);
            assert_eq!(got, want, "{pc:#x}");
            assert_eq!(
                got.is_fatal(),
                matches!(want, UnwindPc::HelperIsland | UnwindPc::NativeFrame)
            );
        }
        // Region ends are exclusive.
        assert_eq!(
            classify(0x1_0000_1014, &layout, exec),
            UnwindPc::NativeFrame
        );
        assert_eq!(classify(0x70_05b1_bfff, &layout, exec), UnwindPc::Guest);
    }
}
