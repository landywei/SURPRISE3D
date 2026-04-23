#!/usr/bin/env bash
# Zero-shot eval on a small scene allowlist (scripts/trial_scenes.txt by default via small YAML).
#
# Artifacts (under lavis/<output_dir>/<job_id>/ from the YAML, e.g. lavis/output/reason3d_surprise_zeroshot_small/<timestamp>/):
#   qualitative/predictions.jsonl — one JSON per line: scene_id, ann_id, object_id, text_input, IoUs, mask_npz path
#   qualitative/masks/<scene_id>_<ann_id>.npz — pred_pmask, gt_pmask (float16, per-point)
# Disable saving: REASON3D_SAVE_PREDS=0 or --options run.save_eval_predictions=false
#
# Visualize qualitative/ (PLY + IoU histogram): see scripts/visualize_qualitative_preds.py
# and scripts/run_visualize_qualitative.sh (set QUAL_DIR and REASON3D_PTS_ROOT).
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
if [[ "${REASON3D_SAVE_PREDS:-1}" == "0" ]]; then
  OPTS+=( "run.save_eval_predictions=false" )
fi

exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
