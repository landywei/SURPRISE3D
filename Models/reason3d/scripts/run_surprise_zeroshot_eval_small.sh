#!/usr/bin/env bash
# Zero-shot Reason3D eval on a small scene allowlist (default scripts/trial_scenes.txt).
#
# Same conventions as run_surprise_zeroshot_eval_100.sh; this script just
# overrides the dataset's eval scene allowlist instead of swapping the test
# annotation JSON, so eval still runs on the full Surprise val *.json but is
# limited to the rows whose scene_id is in the allowlist file.
# CFG selects the dataset builder (bare / geo / chain / chainv3 / chainv3_cot)
# just like the 100-row / full-eval scripts.
#
# Variants (set CFG= to match your checkpoint):
#   bare:        lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml (default)
#   geo:         lavis/projects/reason3d/val/reason3d_surprise_zeroshot_geo.yaml
#   chain:       lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chain.yaml
#   chainv3:     lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml
#   chainv3_cot: lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3_cot.yaml
#                (model arch reason3d_t5_chainv3_cot, builder 3d_refer_chainv3_cot;
#                 two-pass predict_seg also persists the pass-1 intermediate mask
#                 alongside pred/gt when REASON3D_SAVE_EVAL_MASKS=1)
#
# Quick chainv3 run on the committed 3-scene allowlist:
#   CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
#     REASON3D_CKPT=/path/to/ckpt.pth bash scripts/run_surprise_zeroshot_eval_small.sh
#
# Override the allowlist file (e.g. another N-scene file you committed):
#   SURPRISE_SCENE_ALLOWLIST=/abs/path/to/scenes.txt bash scripts/run_surprise_zeroshot_eval_small.sh
#
# Save qualitative artifacts (small allowlists are cheap; masks remain large
# per row but feasible for a handful of scenes):
#   REASON3D_SAVE_PREDS=1                    # qualitative/predictions.jsonl
#   REASON3D_SAVE_EVAL_MASKS=1               # also qualitative/masks/*.npz
#                                            # (for chainv3_cot, each .npz also
#                                            #  contains pred_pmask_intermediate
#                                            #  on rows where the two-pass branch
#                                            #  fired -- see ``intermediate_in_npz``
#                                            #  in predictions.jsonl)
#   REASON3D_SAVE_PREDS=0                    # force no JSONL even if YAML true
#
# Multi-GPU eval (rarely needed for a few scenes):
#   NPROC=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 bash scripts/run_surprise_zeroshot_eval_small.sh
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
# Default scene allowlist: committed 3-scene trial list.
SCENES_DEFAULT="${REASON3D}/scripts/trial_scenes.txt"
SCENES="${SURPRISE_SCENE_ALLOWLIST:-$SCENES_DEFAULT}"

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
if [[ ! -f "$SCENES" ]]; then
  echo "Scene allowlist file not found: $SCENES" >&2
  echo "Either commit the trial scenes to ${SCENES_DEFAULT} or set SURPRISE_SCENE_ALLOWLIST=/abs/path.txt." >&2
  exit 1
fi

# Resolve to absolute path so evaluate.py reads it regardless of cwd / output dir.
SCENES="$(cd "$(dirname "$SCENES")" && pwd)/$(basename "$SCENES")"

_dataset_key_for_cfg() {
  case "$1" in
    *zeroshot_geo.yaml|*small_geo.yaml) echo "3d_refer_geo" ;;
    *zeroshot_chain.yaml|*small_chain.yaml) echo "3d_refer_chain" ;;
    # NOTE: match *_cot before plain chainv3 so the more specific CoT YAML
    # gets the chainv3_cot dataset key.
    *zeroshot_chainv3_cot.yaml) echo "3d_refer_chainv3_cot" ;;
    *zeroshot_chainv3.yaml) echo "3d_refer_chainv3" ;;
    *) echo "3d_refer" ;;
  esac
}

DKEY="$(_dataset_key_for_cfg "$CFG")"

OPTS=(
  "model.reason3d_checkpoint=${CKPT}"
  "datasets.${DKEY}.dataset_init.eval_scene_allowlist_file=${SCENES}"
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
echo "Scene allowlist: $SCENES" >&2

NPROC="${NPROC:-1}"
# evaluate.py + init_distributed_mode: YAML distributed:false exits before reading RANK; must override.
if [[ "$NPROC" -gt 1 ]]; then
  OPTS+=( "run.distributed=true" "run.use_dist_eval_sampler=true" )
  exec torchrun --nproc_per_node="$NPROC" evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
else
  exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
fi
