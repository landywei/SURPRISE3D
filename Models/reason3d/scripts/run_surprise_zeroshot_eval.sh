#!/usr/bin/env bash
# Zero-shot Reason3D eval on Surprise val JSON; skips annotations with no GT instance in .pth.
# Requires pointgroup_ops: run scripts/build_pointgroup_ops.sh once per conda env.
#
# Variants (set CFG= to match your checkpoint):
#   bare:  lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml (default)
#   geo:   lavis/projects/reason3d/val/reason3d_surprise_zeroshot_geo.yaml
#   chain: lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chain.yaml
#
# JSONL without mask .npz (default when saving preds from this script):
#   REASON3D_SAVE_PREDS=1   # qualitative/predictions.jsonl
#   REASON3D_SAVE_EVAL_MASKS=1   # also write qualitative/masks/*.npz (large on full val)
#   REASON3D_SAVE_PREDS=0   # force no JSONL even if YAML has true
#
# Multi-GPU eval (shard test set; requires torchrun + NCCL):
#   NPROC=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 bash scripts/run_surprise_zeroshot_eval.sh
#
# Point cloud .pth dir (default YAML uses pth_rel_subdir=processed under points.storage):
#   REASON3D_PTH_SUBDIR=processed_surprise_full_pth
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp   # omit if same as YAML
# filter_missing_gt_in_pth is forced ON here (drops QA rows with no object_id in .pth). Opt out:
#   REASON3D_FILTER_MISSING_GT_IN_PTH=0
#
# Resume after crash (reuse the same output job folder name under lavis/<output_dir>/):
#   REASON3D_EVAL_RESUME=1 REASON3D_EVAL_JOB_ID=<timestamp folder>
#   Same CFG and run.output_dir in YAML as the partial run. Use REASON3D_SAVE_PREDS=1 (or YAML save_eval_predictions true).
#   With multi-GPU + DistributedSampler, prefer resuming single-GPU or verify sampler length matches filtered dataset.
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

# srun/Slurm shells often lack conda on PATH (same as run_surprise_zeroshot_eval_small.sh).
# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml}"
CKPT="${REASON3D_CKPT:-}"

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

_append_save_pred_opts() {
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
}

OPTS=( "model.reason3d_checkpoint=${CKPT}" )
_append_save_pred_opts
if [[ -n "${REASON3D_PTH_SUBDIR:-}" ]]; then
  case "$CFG" in
    *zeroshot_geo.yaml)
      OPTS+=( "datasets.3d_refer_geo.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
      ;;
    *zeroshot_chain.yaml|*small_chain.yaml)
      OPTS+=( "datasets.3d_refer_chain.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
      ;;
    *)
      OPTS+=( "datasets.3d_refer.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
      ;;
  esac
fi
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  case "$CFG" in
    *zeroshot_geo.yaml)
      OPTS+=( "datasets.3d_refer_geo.build_info.points.storage=${REASON3D_PTS_ROOT}" )
      ;;
    *zeroshot_chain.yaml|*small_chain.yaml)
      OPTS+=( "datasets.3d_refer_chain.build_info.points.storage=${REASON3D_PTS_ROOT}" )
      ;;
    *)
      OPTS+=( "datasets.3d_refer.build_info.points.storage=${REASON3D_PTS_ROOT}" )
      ;;
  esac
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  case "$CFG" in
    *zeroshot_geo.yaml)
      OPTS+=( "datasets.3d_refer_geo.dataset_init.filter_missing_gt_in_pth=false" )
      ;;
    *zeroshot_chain.yaml|*small_chain.yaml)
      OPTS+=( "datasets.3d_refer_chain.dataset_init.filter_missing_gt_in_pth=false" )
      ;;
    *)
      OPTS+=( "datasets.3d_refer.dataset_init.filter_missing_gt_in_pth=false" )
      ;;
  esac
else
  case "$CFG" in
    *zeroshot_geo.yaml)
      OPTS+=( "datasets.3d_refer_geo.dataset_init.filter_missing_gt_in_pth=true" )
      ;;
    *zeroshot_chain.yaml|*small_chain.yaml)
      OPTS+=( "datasets.3d_refer_chain.dataset_init.filter_missing_gt_in_pth=true" )
      ;;
    *)
      OPTS+=( "datasets.3d_refer.dataset_init.filter_missing_gt_in_pth=true" )
      ;;
  esac
fi
if [[ "${REASON3D_EVAL_RESUME:-0}" == "1" ]]; then
  if [[ -z "${REASON3D_EVAL_JOB_ID:-}" ]]; then
    echo "REASON3D_EVAL_RESUME=1 requires REASON3D_EVAL_JOB_ID to the interrupted run id (see lavis/output/<cfg>/<id>/)." >&2
    exit 1
  fi
  OPTS+=( "run.eval_resume_predictions=true" "run.save_eval_predictions=true" )
  if [[ "${REASON3D_SAVE_EVAL_MASKS:-0}" == "1" ]]; then
    OPTS+=( "run.save_eval_prediction_masks=true" )
  else
    OPTS+=( "run.save_eval_prediction_masks=false" )
  fi
fi

NPROC="${NPROC:-1}"
# evaluate.py + init_distributed_mode: YAML distributed:false exits before reading RANK; must override.
if [[ "$NPROC" -gt 1 ]]; then
  OPTS+=( "run.distributed=true" "run.use_dist_eval_sampler=true" )
  exec torchrun --nproc_per_node="$NPROC" evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
else
  exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
fi
