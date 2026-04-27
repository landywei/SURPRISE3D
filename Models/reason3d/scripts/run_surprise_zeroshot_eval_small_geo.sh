#!/usr/bin/env bash
# Zero-shot / checkpoint eval for reason3d_t5_geo + 3d_refer_geo (small allowlist + eval_max_samples in YAML).
#
# model.geo_relational_cfg in the YAML must match training (e.g. hidden_dim 96 for geo_quick, 128 for full geo).
# Override: --options model.geo_relational_cfg.hidden_dim=96
#
# From Models/reason3d (conda env, PYTHONPATH=., pointgroup_ops built):
#   REASON3D_CKPT=/path/to/ckpt_epoch_X.pth bash scripts/run_surprise_zeroshot_eval_small_geo.sh
#
# Override scenes or cap:
#   CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small_geo.yaml \
#   REASON3D_CKPT=/path/to/ckpt.pth \
#   REASON3D_EVAL_OPTIONS="datasets.3d_refer_geo.dataset_init.eval_max_samples=64"
# Resume: REASON3D_EVAL_RESUME=1 REASON3D_EVAL_JOB_ID=<id> (and REASON3D_SAVE_PREDS=1 if disabled in YAML)
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

CFG="${CFG:-lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small_geo.yaml}"
CKPT="${REASON3D_CKPT:-}"

if [[ -z "$CKPT" ]]; then
  echo "Set REASON3D_CKPT to your geo finetune checkpoint (.pth with checkpoint[\"model\"])." >&2
  echo "Example: REASON3D_CKPT=/path/to/ckpt_epoch_1.pth bash $0" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "REASON3D_CKPT is not a readable file: $CKPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

OPTS=( "model.reason3d_checkpoint=${CKPT}" )
if [[ "${REASON3D_SAVE_PREDS:-1}" == "0" ]]; then
  OPTS+=( "run.save_eval_predictions=false" )
fi
if [[ -n "${REASON3D_EVAL_OPTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  OPTS+=( ${REASON3D_EVAL_OPTIONS} )
fi
if [[ "${REASON3D_EVAL_RESUME:-0}" == "1" ]]; then
  if [[ -z "${REASON3D_EVAL_JOB_ID:-}" ]]; then
    echo "REASON3D_EVAL_RESUME=1 requires REASON3D_EVAL_JOB_ID (folder under lavis/output/<run>/)." >&2
    exit 1
  fi
  OPTS+=( "run.eval_resume_predictions=true" "run.save_eval_predictions=true" )
fi

exec python evaluate.py --cfg-path "$CFG" --options "${OPTS[@]}"
