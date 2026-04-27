#!/usr/bin/env bash
# Small zero-shot eval using builder ``3d_refer_chain`` (same scenes as trial_scenes.txt).
# Mirrors run_surprise_zeroshot_eval_small.sh; options use ``datasets.3d_refer_chain``.
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small_chain.yaml}"
CKPT="${REASON3D_CKPT:-}"

if [[ -z "$CKPT" ]]; then
  echo "Set REASON3D_CKPT to the full Reason3D checkpoint (.pth), e.g.:" >&2
  echo "  REASON3D_CKPT=/path/to/reason3d.pth bash $0" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "REASON3D_CKPT is not a readable file: $CKPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

OPTS=( "model.reason3d_checkpoint=${CKPT}" )
if [[ "${REASON3D_SAVE_PREDS:-}" == "1" ]]; then
  OPTS+=( "run.save_eval_predictions=true" )
elif [[ "${REASON3D_SAVE_PREDS:-}" == "0" ]]; then
  OPTS+=( "run.save_eval_predictions=false" )
fi
REASON3D_PTH_SUBDIR="${REASON3D_PTH_SUBDIR:-processed_surprise_full_pth}"
OPTS+=( "datasets.3d_refer_chain.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  OPTS+=( "datasets.3d_refer_chain.build_info.points.storage=${REASON3D_PTS_ROOT}" )
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  OPTS+=( "datasets.3d_refer_chain.dataset_init.filter_missing_gt_in_pth=false" )
else
  OPTS+=( "datasets.3d_refer_chain.dataset_init.filter_missing_gt_in_pth=true" )
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
