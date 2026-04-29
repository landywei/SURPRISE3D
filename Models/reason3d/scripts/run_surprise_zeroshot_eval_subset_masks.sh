#!/usr/bin/env bash
# Rerun zero-shot eval on a *subset* Surprise val JSON (e.g. 100 rows from export_surprise_val_subset.py)
# with qualitative JSONL **and** mask .npz files enabled.
#
# Prereq: build the subset JSON once:
#   python scripts/sample_surprise_predictions.py ... --out-dir /path/to/sample_dir
#   python scripts/export_surprise_val_subset.py \
#     --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \
#     --manifest /path/to/sample_dir/sample_manifest.json \
#     --out /path/to/surprise_val_100.json
#
# Then (bare example):
#   cd Models/reason3d && . scripts/conda_init_reason3d.sh
#   SURPRISE_VAL_SUBSET=/path/to/surprise_val_100.json \
#   REASON3D_CKPT=/path/to/reason3d.pth \
#   REASON3D_SAVE_PREDS=1 REASON3D_SAVE_EVAL_MASKS=1 \
#   bash scripts/run_surprise_zeroshot_eval_subset_masks.sh
#
# Geo / chain: set CFG= to reason3d_surprise_zeroshot_geo.yaml or reason3d_surprise_zeroshot_chain.yaml
# and override the matching datasets.*.build_info.annotations.test.storage key (handled below).
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"
# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml}"
CKPT="${REASON3D_CKPT:-}"
SUB="${SURPRISE_VAL_SUBSET:-}"

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Set REASON3D_CKPT to a readable .pth" >&2
  exit 1
fi
if [[ -z "$SUB" || ! -f "$SUB" ]]; then
  echo "Set SURPRISE_VAL_SUBSET to the JSON list from export_surprise_val_subset.py" >&2
  exit 1
fi

OPTS=( "model.reason3d_checkpoint=${CKPT}" "run.save_eval_predictions=true" "run.save_eval_prediction_masks=true" )

case "$CFG" in
  *zeroshot_geo.yaml)
    OPTS+=( "datasets.3d_refer_geo.build_info.annotations.test.storage=${SUB}" )
    ;;
  *zeroshot_chain.yaml|*small_chain.yaml)
    OPTS+=( "datasets.3d_refer_chain.build_info.annotations.test.storage=${SUB}" )
    ;;
  *)
    OPTS+=( "datasets.3d_refer.build_info.annotations.test.storage=${SUB}" )
    ;;
esac

# Optional same overrides as run_surprise_zeroshot_eval.sh
if [[ -n "${REASON3D_PTH_SUBDIR:-}" ]]; then
  case "$CFG" in
    *zeroshot_geo.yaml) OPTS+=( "datasets.3d_refer_geo.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" ) ;;
    *zeroshot_chain.yaml|*small_chain.yaml) OPTS+=( "datasets.3d_refer_chain.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" ) ;;
    *) OPTS+=( "datasets.3d_refer.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" ) ;;
  esac
fi
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  case "$CFG" in
    *zeroshot_geo.yaml) OPTS+=( "datasets.3d_refer_geo.build_info.points.storage=${REASON3D_PTS_ROOT}" ) ;;
    *zeroshot_chain.yaml|*small_chain.yaml) OPTS+=( "datasets.3d_refer_chain.build_info.points.storage=${REASON3D_PTS_ROOT}" ) ;;
    *) OPTS+=( "datasets.3d_refer.build_info.points.storage=${REASON3D_PTS_ROOT}" ) ;;
  esac
fi

echo "CFG=$CFG" >&2
echo "Subset annotations: $SUB" >&2
echo "Masks + predictions will be written under the run output_dir qualitative/ (can be large)." >&2

exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
