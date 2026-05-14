#!/usr/bin/env bash
# Materialize the canonical *filtered* Surprise val JSON + instance-id cache.
#
# Run once before launching the 8-checkpoint sweep. After this, every val YAML
# under lavis/projects/reason3d/val/ should have:
#
#   datasets:
#     <key>:
#       dataset_init:
#         filter_missing_gt_in_pth: false
#       build_info:
#         annotations:
#           test:
#             storage: $OUT_JSON
#
# so each run sees a byte-identical row set regardless of .pth state, eval
# auto-resume, or which gpu / shell / day the run is launched from.
#
# Defaults (override via env):
#   VAL_JSON   : /nfs-stor/lan.wei/data/annotations/surprise_val.json
#   PTS_ROOT   : /nfs-stor/lan.wei/data/scannetpp
#   PTH_SUBDIR : processed_surprise_full_pth
#   OUT_JSON   : /nfs-stor/lan.wei/data/annotations/surprise_val_filtered_v1.json
#   OUT_CACHE  : /nfs-stor/lan.wei/data/annotations/surprise_inst_id_cache_v1.json

set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh" 2>/dev/null || true

VAL_JSON="${VAL_JSON:-/nfs-stor/lan.wei/data/annotations/surprise_val.json}"
PTS_ROOT="${PTS_ROOT:-/nfs-stor/lan.wei/data/scannetpp}"
PTH_SUBDIR="${PTH_SUBDIR:-processed_surprise_full_pth}"
OUT_JSON="${OUT_JSON:-/nfs-stor/lan.wei/data/annotations/surprise_val_filtered_v1.json}"
OUT_CACHE="${OUT_CACHE:-/nfs-stor/lan.wei/data/annotations/surprise_inst_id_cache_v1.json}"

echo "Building filtered val JSON:"
echo "  --val-json   $VAL_JSON"
echo "  --pts-root   $PTS_ROOT"
echo "  --pth-subdir $PTH_SUBDIR"
echo "  --out-json   $OUT_JSON"
echo "  --out-cache  $OUT_CACHE"

python3 scripts/build_filtered_surprise_val.py \
  --val-json   "$VAL_JSON" \
  --pts-root   "$PTS_ROOT" \
  --pth-subdir "$PTH_SUBDIR" \
  --out-json   "$OUT_JSON" \
  --out-cache  "$OUT_CACHE"

echo "Done. Update the val YAMLs to point at $OUT_JSON and set"
echo "  dataset_init.filter_missing_gt_in_pth: false"
