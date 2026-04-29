#!/usr/bin/env bash
# Export PLYs only for row indices listed in ROW_INDICES_FILE (e.g. row_indices_bare.txt
# from sample_surprise_predictions.py). Requires mask .npz per row (enable save_eval_prediction_masks).
#
#   cd Models/reason3d && export PYTHONPATH=.
#   QUAL_DIR=lavis/output/reason3d_surprise_zeroshot/20260427161/qualitative \
#   ROW_INDICES_FILE=lavis/output/_samples/surprise_100/row_indices_bare.txt \
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp \
#   REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
#   bash scripts/run_visualize_qualitative_subset.sh --stride 5 --out-dir /tmp/vis100_bare
set -euo pipefail
REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

QUAL_DIR="${QUAL_DIR:-}"
ROW_INDICES_FILE="${ROW_INDICES_FILE:-}"
if [[ -z "$QUAL_DIR" || -z "$ROW_INDICES_FILE" ]]; then
  echo "Set QUAL_DIR (…/qualitative) and ROW_INDICES_FILE (e.g. row_indices_bare.txt)." >&2
  exit 1
fi

exec python scripts/visualize_qualitative_preds.py \
  --qual-dir "$QUAL_DIR" \
  --pts-root "${REASON3D_PTS_ROOT:-}" \
  --pth-subdir "${REASON3D_PTH_SUBDIR:-processed}" \
  --row-indices-file "$ROW_INDICES_FILE" \
  "$@"
