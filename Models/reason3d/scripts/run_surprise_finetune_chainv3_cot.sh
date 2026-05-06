#!/usr/bin/env bash
# Finetune Reason3D with chain v3 CoT (multi-step reasoning + W1-pure mass-pool
# feedback + CriterionV3 loss). Architecture is `Reason3DT5ChainV3CoT`, which
# adds a single linear `mask_pool_proj` (d_sp -> d_t5) on top of
# `Reason3DT5ChainV3`; everything else is inherited.
#
# Builder: `3d_refer_chainv3_cot`. Task: `3d_refer_seg_v3` (per-instance
# `meanMaxIoU` / `hit@tau` metrics; same as chain v3 loss-only).
#
# Same env conventions as run_surprise_finetune_chainv3.sh; only the dataset
# prefix and config path change.
#
# Common knobs (forward to `REASON3D_TRAIN_OPTIONS`):
#   datasets.3d_refer_chainv3_cot.dataset_init.cot_template_prob=0.5
#   model.seg_criterion_cfg.enable_boundary=true
#   model.seg_criterion_cfg.enable_point_aux=true
#
# Resume from a mid-epoch checkpoint:
#   REASON3D_RESUME_CKPT=output/.../checkpoint_ep0003_iter00002000.pth ...
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

CFG="${CFG:-lavis/projects/reason3d/train/reason3d_surprise_finetune_chainv3_cot.yaml}"
INIT_CKPT="${REASON3D_INIT_CKPT:-}"
if [[ -z "$INIT_CKPT" && -n "${REASON3D_CKPT:-}" ]]; then
  INIT_CKPT="$REASON3D_CKPT"
fi

if [[ -z "$INIT_CKPT" ]]; then
  echo "Set REASON3D_INIT_CKPT to the Reason3D .pth to finetune from (weights with checkpoint[\"model\"])." >&2
  echo "Example: REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d.pth bash $0" >&2
  exit 1
fi
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "REASON3D_INIT_CKPT is not a readable file: $INIT_CKPT" >&2
  exit 1
fi

OPTS=( "model.reason3d_checkpoint=${INIT_CKPT}" )

if [[ -n "${REASON3D_RESUME_CKPT:-}" ]]; then
  if [[ ! -f "$REASON3D_RESUME_CKPT" ]]; then
    echo "REASON3D_RESUME_CKPT is not a readable file: $REASON3D_RESUME_CKPT" >&2
    exit 1
  fi
  OPTS+=( "run.resume_ckpt_path=${REASON3D_RESUME_CKPT}" )
fi

REASON3D_PTH_SUBDIR="${REASON3D_PTH_SUBDIR:-processed_surprise_full_pth}"
OPTS+=( "datasets.3d_refer_chainv3_cot.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  OPTS+=( "datasets.3d_refer_chainv3_cot.build_info.points.storage=${REASON3D_PTS_ROOT}" )
fi
if [[ -n "${REASON3D_INSTANCE_CACHE_FILE:-}" ]]; then
  OPTS+=( "datasets.3d_refer_chainv3_cot.dataset_init.instance_id_cache_file=${REASON3D_INSTANCE_CACHE_FILE}" )
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  OPTS+=( "datasets.3d_refer_chainv3_cot.dataset_init.filter_missing_gt_in_pth=false" )
else
  OPTS+=( "datasets.3d_refer_chainv3_cot.dataset_init.filter_missing_gt_in_pth=true" )
fi

if [[ -n "${REASON3D_TRAIN_OPTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  OPTS+=( ${REASON3D_TRAIN_OPTIONS} )
fi

NPROC="${NPROC:-1}"
if [[ "$NPROC" -gt 1 ]]; then
  OPTS+=( "run.distributed=true" )
fi

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${_LIB_DIR}/conda_init_reason3d.sh"
unset _LIB_DIR

if [[ "$NPROC" -gt 1 ]]; then
  exec torchrun --nproc_per_node="$NPROC" train.py --cfg-path "$CFG" --options "${OPTS[@]}"
else
  exec python train.py --cfg-path "$CFG" --options "${OPTS[@]}"
fi
