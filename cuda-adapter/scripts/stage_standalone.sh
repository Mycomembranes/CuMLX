#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# stage_standalone.sh - assemble a standalone, publishable tree of the CUDA
# device adapter in a NEW directory with its OWN fresh git history.
#
#   stage_standalone.sh [--with-ptxas] <destination directory>      (must not exist)
#
# What goes in, all taken from the committed HEAD of this repository (never
# from uncommitted working files, and never this repository's history):
#   portkit/cuda-adapter/        the adapter
#   crates/cuda-driverkit/       src, include, README, manifest (the adapter
#                                server's card backend links it)
#   three test fixtures          vecadd_macos.{cubin,ptx}, matmul_macos.cubin
#   integration/                 the fake-device end-to-end test (it runs only
#                                inside the full cuda-oxide tree)
#   LICENSE-APACHE, LICENSE-NVIDIA, rust-toolchain.toml
# and what is generated here: a two-member workspace manifest, README.md,
# PUBLISH_CHECKLIST.md, .gitignore.
#
# --with-ptxas also adds the native PTX assembler port (portkit/ptxas-native,
# portkit/binary-ko-loader, tools/ptxas-resolve.sh, tools/ptxas-darwin-arm64,
# tools/fetch-cuda-redist.sh). Those files are not all committed in the source
# repository yet, so they are copied from the WORKING TREE, honouring its
# ignore rules. The port contains no NVIDIA bytes: NVIDIA's
# libnvptxcompiler_static.a is fetched by the user from NVIDIA and everything
# derived from it lives outside the tree.
#
# Left out on purpose: this repository's git history (it contains signing keys
# in its first commit), the dext source, firmware, the serve scripts, ptxas,
# run outputs, and cuda-driverkit's tests/ and examples/ (they need crates that
# are not part of this tree).
#
# Nothing is pushed anywhere. The result is a local directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WITH_PTXAS=0
if [ "${1:-}" = "--with-ptxas" ]; then
    WITH_PTXAS=1
    shift
fi
DEST="${1:?usage: stage_standalone.sh [--with-ptxas] <destination directory>}"

if [ -e "$DEST" ]; then
    echo "destination already exists, refusing to overwrite: $DEST" >&2
    exit 2
fi
cd "$REPO_ROOT"
SOURCE_COMMIT="$(git rev-parse HEAD)"
if ! git diff --quiet HEAD -- portkit/cuda-adapter; then
    echo "portkit/cuda-adapter has uncommitted changes; commit them first (the stage is built from HEAD)" >&2
    exit 2
fi

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
echo "[stage] exporting from commit $SOURCE_COMMIT"
git archive HEAD portkit/cuda-adapter crates/cuda-driverkit/src crates/cuda-driverkit/include \
    crates/cuda-driverkit/Cargo.toml crates/cuda-driverkit/README.md crates/cuda-driverkit/cbindgen.toml \
    LICENSE-APACHE LICENSE-NVIDIA rust-toolchain.toml | tar -x -C "$DEST"
mkdir -p "$DEST/integration"
git show HEAD:scripts/tests/test_cuda_adapter_shared_driver.py > "$DEST/integration/test_cuda_adapter_shared_driver.py"

# Fixtures (not tracked in the source repository; built there from its own kernels).
ASSETS="$REPO_ROOT/crates/cuda-driverkit/test_assets"
mkdir -p "$DEST/crates/cuda-driverkit/test_assets/cubins" "$DEST/crates/cuda-driverkit/test_assets/kernels"
for f in cubins/vecadd_macos.cubin cubins/matmul_macos.cubin kernels/vecadd_macos.ptx; do
    if [ -f "$ASSETS/$f" ]; then
        cp "$ASSETS/$f" "$DEST/crates/cuda-driverkit/test_assets/$f"
    else
        echo "[stage] WARNING: fixture $f is missing; the self-test needs it" >&2
    fi
done

PTXAS_MEMBER=""
PTXAS_EXCLUDE=""
PTXAS_ROW=""
PTXAS_BUILD_NOTE="Steps that need pieces outside this tree are skipped and say so: the PTX path (needs cuda-oxide's
native \`ptxas\`) and the hybrid_ops client."
if [ "$WITH_PTXAS" = 1 ]; then
    echo "[stage] adding the native ptxas port from the working tree"
    git ls-files --cached --others --exclude-standard -- portkit/ptxas-native portkit/binary-ko-loader \
        tools/ptxas-resolve.sh tools/ptxas-darwin-arm64 tools/fetch-cuda-redist.sh \
        | grep -v -e '__pycache__' -e '\.DS_Store$' -e '/target/' | while IFS= read -r f; do
            [ -f "$f" ] || continue
            mkdir -p "$DEST/$(dirname "$f")"
            cp -p "$f" "$DEST/$f"
        done
    PTXAS_MEMBER='    "portkit/binary-ko-loader",'
    PTXAS_EXCLUDE='exclude = ["portkit/ptxas-native"]'
    PTXAS_ROW='| \`portkit/ptxas-native/\`, \`portkit/binary-ko-loader/\`, \`tools/\` | the native PTX assembler port: loads NVIDIA'"'"'s Linux-aarch64 \`libnvptxcompiler_static.a\` in-process on macOS. NVIDIA'"'"'s archive is NOT included; see \`portkit/ptxas-native/README.md\` |'
    PTXAS_BUILD_NOTE="The PTX path is skipped until the native \`ptxas\` is built (\`portkit/ptxas-native/build.sh\`, which
needs NVIDIA's archive fetched with \`tools/fetch-cuda-redist.sh\`). The hybrid_ops client is skipped unless its
source is present."
fi

# cuda-driverkit's dev-dependencies point at crates that are not in this tree,
# and its tests/ and examples/ are not exported, so drop that table.
python3 - "$DEST/crates/cuda-driverkit/Cargo.toml" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()
start = text.index("[target.'cfg(target_os = \"macos\")'.dev-dependencies]")
rest = text[start:]
match = re.search(r"\n\[", rest[1:])
end = start + 1 + match.start() + 1 if match else len(text)
note = ("# The standalone tree omits this crate's tests/ and examples/ and the two\n"
        "# workspace crates they depend on, so its dev-dependencies are omitted too.\n")
open(path, "w").write(text[:start] + note + text[end:])
PY

cat > "$DEST/Cargo.toml" <<EOF
# Standalone workspace for the CUDA device adapter (generated by
# portkit/cuda-adapter/scripts/stage_standalone.sh). Package and dependency
# values mirror the cuda-oxide workspace the two crates come from.
[workspace]
resolver = "3"
members = [
    "crates/cuda-driverkit",
    "portkit/cuda-adapter/server",
$PTXAS_MEMBER
]
$PTXAS_EXCLUDE

[workspace.package]
version = "0.1.0"
edition = "2024"
authors = ["Mukshud Ahamed"]
license = "Apache-2.0"
repository = "https://github.com/NVlabs/cuda-oxide.git"

[workspace.dependencies]
cuda-driverkit = { path = "crates/cuda-driverkit" }
futures = "0.3"
thiserror = "2"
anyhow = "1.0"
once_cell = "1.18"
log = "0.4"
libloading = "0.8"
EOF

cat > "$DEST/.gitignore" <<'EOF'
target/
portkit/cuda-adapter/build/
portkit/cuda-adapter/shim/generated/
__pycache__/
.DS_Store
EOF

cat > "$DEST/README.md" <<EOF
# CUDA device adapter for macOS (legacy CUDA Driver API clients on a Thunderbolt eGPU)

A stand-in \`libcuda\` (x86_64 for Rosetta clients, arm64 native) that forwards the classic CUDA Driver
API to an adapter server, which reaches an NVIDIA card served by the cuda-oxide macOS DriverKit port.
NVIDIA's last macOS CUDA driver (418.105, x86_64 only) cannot work on Apple Silicon; this replaces the
library boundary and leaves applications and NVIDIA's installed files untouched.

**Start here:** [\`portkit/cuda-adapter/README.md\`](portkit/cuda-adapter/README.md) has the design, the
status table, the limits and the results of the first run on a card.

## What is in this tree

| Path | What |
|---|---|
| \`portkit/cuda-adapter/\` | the adapter: stand-in library, server, runtime and cuBLASLt layers, Fortran bindings, launchers, build script, self-test |
| \`crates/cuda-driverkit/\` | the cuda-oxide macOS driver crate (source and headers only); the server's card backend links it |
| \`crates/cuda-driverkit/test_assets/\` | three small fixtures the self-test uses |
| \`integration/\` | the end-to-end test against the CPU fake device; it needs the full cuda-oxide tree |
$PTXAS_ROW

This is a snapshot exported from a private development repository at commit \`${SOURCE_COMMIT:0:12}\`. It
has a fresh history.

## Build and self-test (macOS on Apple Silicon, no GPU needed)

\`\`\`bash
portkit/cuda-adapter/scripts/build_cuda_adapter.sh
\`\`\`

This builds the server and the stand-in library and runs the self-test on the host-memory \`loopback\`
backend. $PTXAS_BUILD_NOTE

## What you cannot do with this tree alone

Running against a real card needs a served cuda-oxide exact-serve stack (the DriverKit dext, the PCI
server, the launch server), none of which is here. The Rust toolchain is pinned in \`rust-toolchain.toml\`.

## License

Source files carry SPDX headers (Apache-2.0). \`crates/cuda-driverkit/Cargo.toml\` names
\`LICENSE-NVIDIA\` as its license file. See \`PUBLISH_CHECKLIST.md\` before publishing.
EOF

cat > "$DEST/PUBLISH_CHECKLIST.md" <<EOF
# Before this tree is pushed anywhere

Generated $(date -u +%Y-%m-%d) from commit \`$SOURCE_COMMIT\`. Nothing has been uploaded.

- [ ] **License.** \`crates/cuda-driverkit/Cargo.toml\` says \`license-file = "../../LICENSE-NVIDIA"\` while
      its source files carry Apache-2.0 SPDX headers and an NVIDIA copyright line. Decide which applies and
      make the manifest, the headers and the license files agree. The NVIDIA Software License text forbids
      modification and open-source redistribution of the software it covers.
- [ ] **Copyright line.** Every new file follows the source repository's rule and says
      "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES". Confirm that is the owner you intend to state.
- [ ] **Driver internals.** \`crates/cuda-driverkit/src\` contains the firmware/GSP-RM boot path
      (\`gsp_boot.rs\`, \`fsp_cot.rs\`, \`gsp_*.rs\`, \`devinit_allowlist.rs\`, \`ogk_*.rs\`,
      \`gsp_golden_image.rs\`). Review them for anything derived from material you may not redistribute.
- [ ] **Fixtures.** \`crates/cuda-driverkit/test_assets/\` holds two compiled cubins and one PTX file built
      from this project's own kernels. Confirm you want compiled binaries in the repository.
- [ ] **Symbol list.** \`portkit/cuda-adapter/symbols/cuda_10_1.txt\` is the list of function names exported
      by NVIDIA's CUDA 10.1 framework (names only, no code).
- [ ] **Claims.** The README states what ran: small attended workloads on one card. It makes no general
      claim of CUDA support on macOS.
- [ ] **Personal paths.** About 49 lines in 11 files under \`crates/cuda-driverkit/src\` contain absolute
      paths under your home directory (your macOS user name), mostly in comments and default paths.
      List them with \`git grep -n /Users/\` and decide whether to rewrite them.
$( [ "$WITH_PTXAS" = 1 ] && printf '%s\n' "- [ ] **ptxas port.** \`portkit/ptxas-native\` and \`portkit/binary-ko-loader\` were copied from uncommitted working files. They contain no NVIDIA bytes, but the port exists to run NVIDIA's proprietary compiler library, and its README documents how that library was adapted (register rewriting of its machine code at load time). NVIDIA's license terms on that library restrict modification and reverse engineering; decide whether publishing the method is acceptable to you." "- [ ] **Uncommitted source.** The ptxas files are not committed in the source repository; commit them there so this snapshot can be reproduced." )
- [ ] **Secrets.** Run a secret scan on the final tree (the staging script's caller did one; repeat after
      any edit).
- [ ] **Repository URL.** \`Cargo.toml\` still names the upstream \`NVlabs/cuda-oxide\` URL; change it to the
      new repository.
EOF

echo "[stage] tree written to $DEST"
(cd "$DEST" && git init -q -b main && git add -A && git -c commit.gpgsign=false commit -q -s \
    -m "CUDA device adapter for macOS: standalone snapshot" \
    -m "Exported from a private development repository at commit $SOURCE_COMMIT. Fresh history; see PUBLISH_CHECKLIST.md before publishing." \
    && echo "[stage] committed: $(git log --oneline -1)  ($(git ls-files | wc -l | tr -d ' ') files)")
