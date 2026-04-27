#!/usr/bin/env bash
# Build ScanNet++ .pth files with all instance objectIds (see all_instance_classes patch in third_party/scannetpp).
#
# This uses ScanNet++ (https://github.com/scannetpp/scannetpp), NOT ScanNet v2.
#
# Env:
#   SCANNPP_REPO     - default: SURPRISE3D/third_party/scannetpp
#   SCANNETPP_ROOT   - default: /nfs-stor/lan.wei/data/scannetpp (must contain data/<scene_id>/...)
#   LIST_PATH        - default: Models/reason3d/scripts/trial_scenes.txt (override for full runs)
#   CONFIG           - default: Models/reason3d/scripts/scannetpp_prepare_full_instances.yml
#   OUT_DIR          - if set, overrides YAML out_dir (written into a temp config)
#
# Example (trial scenes):
#   bash scripts/run_scannetpp_prepare_full_instances.sh
#
# Example (custom list):
#   LIST_PATH=/path/to/my_scenes.txt bash scripts/run_scannetpp_prepare_full_instances.sh

set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SURPRISE3D="$(cd "$REASON3D/../.." && pwd)"
SCANNPP_REPO="${SCANNPP_REPO:-$SURPRISE3D/third_party/scannetpp}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/nfs-stor/lan.wei/data/scannetpp}"
LIST_PATH="${LIST_PATH:-$REASON3D/scripts/trial_scenes.txt}"
CONFIG="${CONFIG:-$REASON3D/scripts/scannetpp_prepare_full_instances.yml}"

if [[ ! -d "$SCANNPP_REPO/semantic" ]]; then
  echo "Missing ScanNet++ clone at SCANNPP_REPO=$SCANNPP_REPO" >&2
  echo "From repo root run: git clone https://github.com/scannetpp/scannetpp.git third_party/scannetpp" >&2
  exit 1
fi

if [[ ! -f "$LIST_PATH" ]]; then
  echo "LIST_PATH is not a file: $LIST_PATH" >&2
  exit 1
fi

if [[ ! -d "$SCANNETPP_ROOT/data" ]]; then
  echo "SCANNETPP_ROOT has no data/: $SCANNETPP_ROOT" >&2
  exit 1
fi

LIST_ABS="$(readlink -f "$LIST_PATH")"
ROOT_ABS="$(readlink -f "$SCANNETPP_ROOT")"
CFG_SRC="$(readlink -f "$CONFIG")"

TMPYML="$(mktemp --suffix=.yml)"
trap 'rm -f "$TMPYML"' EXIT

python3 - "$CFG_SRC" "$ROOT_ABS/data" "$LIST_ABS" "${OUT_DIR:-}" "$TMPYML" <<'PY'
import pathlib, sys

src, data_root, list_path, out_dir, dst = sys.argv[1:6]
text = pathlib.Path(src).read_text()
lines = []
for line in text.splitlines():
    s = line.strip()
    if s.startswith("data_root:"):
        lines.append(f"  data_root: {data_root}")
    elif s.startswith("list_path:"):
        continue
    elif s.startswith("out_dir:") and out_dir.strip():
        lines.append(f"out_dir: {out_dir.strip()}")
    elif s.startswith("out_dir:") and not out_dir.strip():
        lines.append(line)
    else:
        lines.append(line)

out = []
inserted = False
for line in lines:
    out.append(line)
    if (not inserted) and line.strip() == "data:":
        out.append(f"  list_path: {list_path}")
        inserted = True
if not inserted:
    raise SystemExit("YAML must contain a top-level 'data:' key (see scannetpp_prepare_full_instances.yml)")
pathlib.Path(dst).write_text("\n".join(out) + "\n")
PY

# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh" 2>/dev/null || true

export PYTHONPATH="${SCANNPP_REPO}:${PYTHONPATH:-}"
cd "$SCANNPP_REPO"
echo "Using config $TMPYML"
echo "First 35 lines:"
head -n 35 "$TMPYML"
exec python -m semantic.prep.prepare_training_data "$TMPYML"
