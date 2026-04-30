#!/usr/bin/env bash
# Zero-shot Reason3D eval on the stratified 100-prompt Surprise val subset.
#
# The 100 prompts are produced by:
#   1. scripts/sample_surprise_predictions.py  -- stratified pick across
#      (question_type, scene_id) buckets from a prior predictions.jsonl
#      (default --intersect-keys across bare/geo/chain runs, --seed 42, n=100)
#   2. scripts/export_surprise_val_subset.py   -- materialises the 100 ann
#      rows back into a Surprise-style val JSON
# The committed subset lives at:
#   Models/reason3d/lavis/output/_samples/surprise_val_100.json (100 rows)
#
# Same conventions as run_surprise_zeroshot_eval.sh; this script just
# additionally overrides the dataset's test annotation storage to the
# 100-row JSON so eval runs on those rows only. CFG selects the dataset
# builder (bare / geo / chain / chainv3) just like the full-eval script.
#
# Variants (set CFG= to match your checkpoint):
#   bare:    lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml (default)
#   geo:     lavis/projects/reason3d/val/reason3d_surprise_zeroshot_geo.yaml
#   chain:   lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chain.yaml
#   chainv3: lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml
#
# Quick chainv3 run on the committed 100-sample JSON:
#   CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
#     REASON3D_CKPT=/path/to/ckpt.pth bash scripts/run_surprise_zeroshot_eval_100.sh
#
# Override the subset JSON (e.g. another 100/200-row file you exported):
#   SURPRISE_VAL_SUBSET=/abs/path/to/surprise_val_100.json bash scripts/run_surprise_zeroshot_eval_100.sh
#
# Save qualitative artifacts (JSONL only is small; masks are large but feasible
# for 100 rows):
#   REASON3D_SAVE_PREDS=1                    # qualitative/predictions.jsonl
#   REASON3D_SAVE_EVAL_MASKS=1               # also qualitative/masks/*.npz
#   REASON3D_SAVE_PREDS=0                    # force no JSONL even if YAML true
#
# Multi-GPU eval (rarely needed for 100 rows):
#   NPROC=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 bash scripts/run_surprise_zeroshot_eval_100.sh
#
# .pth dir / pts root overrides (same as run_surprise_zeroshot_eval.sh):
#   REASON3D_PTH_SUBDIR=processed_surprise_full_pth
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp
#   REASON3D_FILTER_MISSING_GT_IN_PTH=0   # opt out of dropping rows without GT in .pth
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

# srun/Slurm shells often lack conda on PATH (same as run_surprise_zeroshot_eval.sh).
# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml}"
CKPT="${REASON3D_CKPT:-}"
# Default subset JSON: committed 100-row stratified sample.
SUB_DEFAULT="${REASON3D}/lavis/output/_samples/surprise_val_100.json"
SUB="${SURPRISE_VAL_SUBSET:-$SUB_DEFAULT}"

if [[ -z "$CKPT" ]]; then
  echo "Set REASON3D_CKPT to the full Reason3D checkpoint (.pth), e.g.:" >&2
  echo "  REASON3D_CKPT=/path/to/reason3d.pth bash $0" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "REASON3D_CKPT is not a readable file: $CKPT" >&2
  echo "Use an absolute path if your cwd is not the directory that contains the checkpoint." >&2
  exit 1
fi
if [[ ! -f "$SUB" ]]; then
  echo "Subset annotations JSON not found: $SUB" >&2
  echo "Either commit the 100-row subset to ${SUB_DEFAULT} or set SURPRISE_VAL_SUBSET=/abs/path.json." >&2
  exit 1
fi

# Resolve to absolute path so evaluate.py reads it regardless of cwd / output dir.
SUB="$(cd "$(dirname "$SUB")" && pwd)/$(basename "$SUB")"

_dataset_key_for_cfg() {
  case "$1" in
    *zeroshot_geo.yaml|*small_geo.yaml) echo "3d_refer_geo" ;;
    *zeroshot_chain.yaml|*small_chain.yaml) echo "3d_refer_chain" ;;
    *zeroshot_chainv3.yaml) echo "3d_refer_chainv3" ;;
    *) echo "3d_refer" ;;
  esac
}

DKEY="$(_dataset_key_for_cfg "$CFG")"

OPTS=(
  "model.reason3d_checkpoint=${CKPT}"
  "datasets.${DKEY}.build_info.annotations.test.storage=${SUB}"
)

if [[ "${REASON3D_SAVE_PREDS:-}" == "1" ]]; then
  OPTS+=( "run.save_eval_predictions=true" )
  if [[ "${REASON3D_SAVE_EVAL_MASKS:-0}" == "1" ]]; then
    OPTS+=( "run.save_eval_prediction_masks=true" )
  else
    OPTS+=( "run.save_eval_prediction_masks=false" )
  fi
elif [[ "${REASON3D_SAVE_PREDS:-}" == "0" ]]; then
  OPTS+=( "run.save_eval_predictions=false" )
fi

if [[ -n "${REASON3D_PTH_SUBDIR:-}" ]]; then
  OPTS+=( "datasets.${DKEY}.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
fi
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  OPTS+=( "datasets.${DKEY}.build_info.points.storage=${REASON3D_PTS_ROOT}" )
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  OPTS+=( "datasets.${DKEY}.dataset_init.filter_missing_gt_in_pth=false" )
else
  OPTS+=( "datasets.${DKEY}.dataset_init.filter_missing_gt_in_pth=true" )
fi

echo "CFG=$CFG" >&2
echo "Dataset key: $DKEY" >&2
echo "Subset annotations (100 rows expected): $SUB" >&2

NPROC="${NPROC:-1}"
# evaluate.py + init_distributed_mode: YAML distributed:false exits before reading RANK; must override.
if [[ "$NPROC" -gt 1 ]]; then
  OPTS+=( "run.distributed=true" "run.use_dist_eval_sampler=true" )
  exec torchrun --nproc_per_node="$NPROC" evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
else
  exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
fi
