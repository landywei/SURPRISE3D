#!/usr/bin/env bash
# Zero-shot eval on a small scene allowlist (scripts/trial_scenes.txt by default via small YAML).
#
# Artifacts (under lavis/<output_dir>/<job_id>/ from the YAML, e.g. lavis/output/reason3d_surprise_zeroshot_small/<timestamp>/):
#   qualitative/predictions.jsonl — one JSON per line: scene_id, ann_id, object_id, text_input, IoUs, mask_npz path
#   qualitative/masks/<scene_id>_<ann_id>_<eval_save_index>.npz — pred_pmask, gt_pmask (unique per JSONL row)
# Qualitative saves: REASON3D_SAVE_PREDS=1 enable, =0 force disable (YAML may default true on small geo).
#
# New preprocessed .pth dir (relative to points.storage in the YAML, default scannetpp root):
#   REASON3D_PTH_SUBDIR=processed_surprise_full_pth
# If .pth files are not under that storage root, set the root explicitly:
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp
# filter_missing_gt_in_pth forced ON (see run_surprise_zeroshot_eval.sh). Opt out: REASON3D_FILTER_MISSING_GT_IN_PTH=0
# Multi-GPU: NPROC=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 bash ...
#
# Visualize qualitative/ (PLY + IoU histogram): see scripts/visualize_qualitative_preds.py
# and scripts/run_visualize_qualitative.sh (set QUAL_DIR and REASON3D_PTS_ROOT).
#
# Resume partial eval: REASON3D_EVAL_RESUME=1 REASON3D_EVAL_JOB_ID=<id> REASON3D_SAVE_PREDS=1 ...
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small.yaml}"
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

# srun/Slurm shells often lack `conda` on PATH; see scripts/conda_init_reason3d.sh
# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

OPTS=( "model.reason3d_checkpoint=${CKPT}" )
if [[ "${REASON3D_SAVE_PREDS:-}" == "1" ]]; then
  OPTS+=( "run.save_eval_predictions=true" )
elif [[ "${REASON3D_SAVE_PREDS:-}" == "0" ]]; then
  OPTS+=( "run.save_eval_predictions=false" )
fi
if [[ -n "${REASON3D_PTH_SUBDIR:-}" ]]; then
  OPTS+=( "datasets.3d_refer.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
fi
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  OPTS+=( "datasets.3d_refer.build_info.points.storage=${REASON3D_PTS_ROOT}" )
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  OPTS+=( "datasets.3d_refer.dataset_init.filter_missing_gt_in_pth=false" )
else
  OPTS+=( "datasets.3d_refer.dataset_init.filter_missing_gt_in_pth=true" )
fi
if [[ "${REASON3D_EVAL_RESUME:-0}" == "1" ]]; then
  if [[ -z "${REASON3D_EVAL_JOB_ID:-}" ]]; then
    echo "REASON3D_EVAL_RESUME=1 requires REASON3D_EVAL_JOB_ID (folder name under lavis/output/<run>/)." >&2
    exit 1
  fi
  OPTS+=( "run.eval_resume_predictions=true" "run.save_eval_predictions=true" )
fi

NPROC="${NPROC:-1}"
if [[ "$NPROC" -gt 1 ]]; then
  OPTS+=( "run.distributed=true" "run.use_dist_eval_sampler=true" )
  exec torchrun --nproc_per_node="$NPROC" evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
else
  exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
fi
