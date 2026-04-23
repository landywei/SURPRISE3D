#!/usr/bin/env bash
# Zero-shot Reason3D eval on Surprise val JSON; skips annotations with no GT instance in .pth.
# Requires pointgroup_ops: run scripts/build_pointgroup_ops.sh once per conda env.
#
# Optional per-sample predictions for qualitative analysis (large on full val):
#   REASON3D_SAVE_PREDS=1 bash scripts/run_surprise_zeroshot_eval.sh
# Writes lavis/<output_dir>/<job_id>/qualitative/{predictions.jsonl,masks/*.npz}
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

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

eval "$(conda shell.bash hook)"
conda activate reason3d310 2>/dev/null || conda activate reason3d 2>/dev/null || true

OPTS=( "model.reason3d_checkpoint=${CKPT}" )
if [[ "${REASON3D_SAVE_PREDS:-0}" == "1" ]]; then
  OPTS+=( "run.save_eval_predictions=true" )
fi

exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
