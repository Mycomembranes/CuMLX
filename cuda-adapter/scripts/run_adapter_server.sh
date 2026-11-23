#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# run_adapter_server.sh - start cuda_adapter_server against ONE served card.
#
# The adapter is an ordinary shared-CUDA client of an exact-serve stack: it is
# pointed at the card's registry file (serve-pids.<id>.json) and reaches the
# card only through that stack's launch endpoint. It opens no DriverKit
# connection, resets nothing and serves nothing itself.
#
# Without --go this prints the plan and the read-only preflight result and
# starts nothing. With --go it runs the server in the foreground (attended):
# stop it with Ctrl-C or SIGTERM BEFORE stopping the stack; it finishes the
# request in flight and then exits.
#
# Usage:
#   run_adapter_server.sh --serve-state /abs/path/serve-pids.<16hex>.json [--socket PATH] [--go]
#
# The stack must have been served with --shared-cuda (reference firmware
# loader, tensor enabled) and be "ready" with a PASS proof receipt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CUDA_ADAPTER_ROOT:-$SCRIPT_DIR/../build}"
SERVE_STATE=""
SOCKET="${CUDA_ADAPTER_SOCKET:-/tmp/cuda-adapter-$(id -u).sock}"
GO=0

while [ $# -gt 0 ]; do
    case "$1" in
        --serve-state) SERVE_STATE="${2:?--serve-state needs a path}"; shift ;;
        --socket) SOCKET="${2:?--socket needs a path}"; shift ;;
        --go) GO=1 ;;
        -h|--help) sed -n '5,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

SERVER="$ROOT/cuda_adapter_server"
[ -x "$SERVER" ] || SERVER="$ROOT/bin/cuda_adapter_server"
if [ ! -x "$SERVER" ]; then
    echo "cuda_adapter_server not found under $ROOT; run scripts/build_cuda_adapter.sh" >&2
    exit 2
fi
if [ -z "$SERVE_STATE" ]; then
    echo "--serve-state is required: the registry of the ONE card to use" >&2
    echo "(serve-pids.<16hex>.json, printed by the serve verb; see scripts/egpu_resident.sh status)" >&2
    exit 2
fi
case "$SERVE_STATE" in /*) ;; *) echo "--serve-state must be an absolute path" >&2; exit 2 ;; esac

# Read-only preflight of the registry: file contents only, no device access.
# The C ABI repeats and extends these checks (owner capability, identity)
# when the server connects; this gives the operator the reason up front.
BOOT_ID="$(sysctl -n kern.bootsessionuuid 2>/dev/null | tr 'A-Z' 'a-z')"
if ! python3 - "$SERVE_STATE" "$BOOT_ID" <<'PY'
import json, os, stat, sys
path, boot = sys.argv[1], sys.argv[2]
problems = []
try:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        problems.append("not a regular file")
    with open(path) as handle:
        root = json.load(handle)
except (OSError, ValueError) as exc:
    print(f"preflight: cannot read {path}: {exc}", file=sys.stderr)
    sys.exit(1)
if root.get("schema") != "cuda-oxide/egpu-serve-pids.v2":
    problems.append(f"schema is {root.get('schema')!r}")
records = root.get("processes")
if not isinstance(records, list) or len(records) != 1:
    problems.append("the registry must hold exactly one card record")
    records = [{}]
record = records[0] if isinstance(records[0], dict) else {}
if record.get("status") != "ready":
    problems.append(f"stack status is {record.get('status')!r}, not 'ready'")
if record.get("sharedCuda") is not True:
    problems.append("the stack was not served with --shared-cuda")
# "reference": the served stack; "native-service": the native owner's launch service.
if record.get("firmwareLoader") not in ("reference", "native-service"):
    problems.append(f"firmware loader is {record.get('firmwareLoader')!r}, not 'reference' or 'native-service'")
receipt = record.get("proofReceipt")
if not isinstance(receipt, dict) or receipt.get("status") != "PASS":
    problems.append("no PASS proof receipt")
launch = (record.get("endpoints") or {}).get("launch", "")
if not str(launch).startswith("unix:///"):
    problems.append("the launch endpoint is not a private AF_UNIX socket")
elif not os.path.exists(launch[len("unix://"):]):
    problems.append(f"launch socket {launch[len('unix://'):]} does not exist")
# The registry spells the boot session "macos:<kern.bootsessionuuid, lowercase>".
state_boot = str(root.get("hostBootSessionID", ""))
if boot and state_boot != f"macos:{boot}":
    problems.append(f"the registry is from boot {state_boot[6:14] or '?'}, this boot is {boot[:8]}: it is void")
print(f"card {record.get('card')}  session {record.get('sessionKey')}  stack {record.get('stackID')}")
print(f"launch endpoint {launch}")
for problem in problems:
    print(f"preflight: {problem}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY
then
    echo "preflight FAILED: this registry does not describe a qualified shared-CUDA stack for this boot." >&2
    exit 3
fi

echo "plan: $SERVER --backend driverkit --serve-state $SERVE_STATE --socket $SOCKET"
echo "clients: CUDA_ADAPTER_SOCKET=$SOCKET scripts/run_cuda_rosetta.sh|run_cuda_native.sh <app>"
if [ "$GO" != 1 ]; then
    echo "preflight PASS. Nothing was started (no --go)."
    exit 0
fi

# Leave the operator's identity pins in place; drop routing variables that
# would point the C ABI somewhere other than this registry.
unset CUDA_OXIDE_PCISERVER_LAUNCH CUDA_OXIDE_DAEMON
exec env CUDA_OXIDE_CUDA_BACKEND=pciserver \
    "$SERVER" --backend driverkit --serve-state "$SERVE_STATE" --socket "$SOCKET"
