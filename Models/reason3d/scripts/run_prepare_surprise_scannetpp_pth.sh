#!/usr/bin/env bash
# End-to-end ScanNet++ -> Reason3D-style .pth (new output dir by default):
#   1) Full-vocabulary semantics + all instance objectIds (all_instance_classes).
#   2) Uniform sampling + sampled_mesh_vertex_idx (point -> nearest mesh vertex).
#   3) Superpoints via mesh segmentator (same family as prepare_data_reason.py).
#
# Does not overwrite existing processed dirs unless you point OUT_DIR at them.
#
# Env:
#   SCANNPP_REPO     - ScanNet++ python root (default: SURPRISE3D/third_party/scannetpp)
#   SCANNETPP_ROOT   - dataset root with data/<scene>/... (default: /nfs-stor/lan.wei/data/scannetpp)
#   LIST_PATH        - scene ids, one per line (default: trial_scenes.txt = 3 scenes for smoke tests)
#   LIST_ALL_SCENES=1 - ignore LIST_PATH; use every directory under SCANNETPP_ROOT/data (like UniDet preprocess)
#   CONFIG           - YAML template (default: scannetpp_surprise_full_pth.yml)
#   OUT_DIR          - if set, overrides YAML out_dir in a temp config
#
# Optional second stage (UniDet3D superpoints instead of / in addition to segmentator):
#   SUPERSCENE_DIR   - parent of per-scene folders containing *superpoints.npy
#                      If set, runs update_superpoints.py (use --force to replace segmentator ids).
# UniDet3D-only superpoints (skip segmentator in phase 1):
#   PREPARE_EXTRA_ARGS=--no-vertex-superpoints
# Parallel scene processing (Linux fork; default 1):
#   NUM_WORKERS=8

set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SURPRISE3D="$(cd "$REASON3D/../.." && pwd)"
SCANNPP_REPO="${SCANNPP_REPO:-$SURPRISE3D/third_party/scannetpp}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/nfs-stor/lan.wei/data/scannetpp}"
LIST_PATH="${LIST_PATH:-$REASON3D/scripts/trial_scenes.txt}"
CONFIG="${CONFIG:-$REASON3D/scripts/scannetpp_surprise_full_pth.yml}"
SUPERSCENE_DIR="${SUPERSCENE_DIR:-}"
SUPERPOINT_FORCE="${SUPERPOINT_FORCE:-0}"
PREPARE_EXTRA_ARGS="${PREPARE_EXTRA_ARGS:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"

if [[ ! -d "$SCANNPP_REPO/semantic" ]]; then
  echo "Missing ScanNet++ repo at SCANNPP_REPO=$SCANNPP_REPO" >&2
  exit 1
fi
if [[ ! -d "$SCANNETPP_ROOT/data" ]]; then
  echo "SCANNETPP_ROOT has no data/: $SCANNETPP_ROOT" >&2
  exit 1
fi

ROOT_DATA_ABS="$(readlink -f "$SCANNETPP_ROOT/data")"
CFG_SRC="$(readlink -f "$CONFIG")"

TMP_LIST=""
if [[ "${LIST_ALL_SCENES:-0}" == "1" ]]; then
  TMP_LIST="$(mktemp --suffix=_all_scene_ids.txt)"
  for d in "$ROOT_DATA_ABS"/*; do
    [[ -d "$d" ]] || continue
    bn="$(basename "$d")"
    [[ "$bn" == .ipynb_checkpoints ]] && continue
    printf '%s\n' "$bn"
  done | sort -u >"$TMP_LIST"
  echo "LIST_ALL_SCENES=1: wrote $(wc -l <"$TMP_LIST") scene ids to $TMP_LIST"
  LIST_ABS="$(readlink -f "$TMP_LIST")"
else
  if [[ ! -f "$LIST_PATH" ]]; then
    echo "LIST_PATH is not a file: $LIST_PATH" >&2
    echo "Tip: use LIST_ALL_SCENES=1 to process every scene under data/, or set LIST_PATH to a full list." >&2
    exit 1
  fi
  LIST_ABS="$(readlink -f "$LIST_PATH")"
fi

TMPYML="$(mktemp --suffix=.yml)"
trap 'rm -f "$TMPYML" ${TMP_LIST:+"$TMP_LIST"}' EXIT

python3 - "$CFG_SRC" "$ROOT_DATA_ABS" "$LIST_ABS" "${OUT_DIR:-}" "$TMPYML" <<'PY'
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
    raise SystemExit("YAML must contain a top-level 'data:' key")
pathlib.Path(dst).write_text("\n".join(out) + "\n")
PY

# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh" 2>/dev/null || true

export PYTHONPATH="${SCANNPP_REPO}:${PYTHONPATH:-}"

echo "=== Phase 1: sample + labels + vertex index + segmentator superpoints ==="
head -n 40 "$TMPYML"
# shellcheck disable=SC2086
python3 "${REASON3D}/scripts/prepare_surprise_scannetpp_pth.py" "$TMPYML" --scannetpp-repo "$SCANNPP_REPO" ${PREPARE_EXTRA_ARGS} --num-workers "${NUM_WORKERS}"

OUT_RESOLVED="$(awk -F': ' '/^out_dir:/{gsub(/^ +| +$/,"",$2); print $2; exit}' "$TMPYML")"

if [[ -n "$SUPERSCENE_DIR" ]]; then
  echo "=== Phase 2: merge UniDet3D superpoints from SUPERSCENE_DIR=$SUPERSCENE_DIR ==="
  FORCE_FLAG=()
  if [[ "$SUPERPOINT_FORCE" == "1" ]]; then
    FORCE_FLAG=(--force)
  fi
  python3 "${REASON3D}/update_superpoints.py" \
    --pth_dir "$OUT_RESOLVED" \
    --scene_dir "$SUPERSCENE_DIR" \
    --scannetpp_root "$SCANNETPP_ROOT" \
    "${FORCE_FLAG[@]:-}"
else
  echo "Skipping UniDet3D superpoint merge (SUPERSCENE_DIR unset). .pth already has segmentator superpoints."
fi

echo "Done. Point Reason3D pth_rel_subdir at: $OUT_RESOLVED"
