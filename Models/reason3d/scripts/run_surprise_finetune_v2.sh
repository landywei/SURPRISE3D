#!/usr/bin/env bash
# Finetune Reason3D on Surprise train JSON (v2: no validation, decimal fix) + ScanNet++ processed .pth (see train/reason3d_surprise_finetune.yaml).
# Default .pth layout: join(points.storage, pth_rel_subdir, "<scene>.pth") — YAML uses processed_surprise_full_pth.
# Override like eval scripts:
#   REASON3D_PTH_SUBDIR=processed_surprise_full_pth
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp
#   REASON3D_INSTANCE_CACHE_FILE=/path/to/.reason3d_instance_ids.json  # optional; should match subdir you load
# filter_missing_gt_in_pth defaults ON in YAML; force here unless REASON3D_FILTER_MISSING_GT_IN_PTH=0
# Requires pointgroup_ops (scripts/build_pointgroup_ops.sh) and a starting Reason3D checkpoint.
#
# Instance-id cache (fast dataset init after first pass): see datasets.3d_refer.dataset_init in the YAML.
#   First run on a cluster: set instance_id_cache_write: true (or override once):
#     REASON3D_TRAIN_OPTIONS="datasets.3d_refer.dataset_init.instance_id_cache_write=true" ...
#   Later runs: instance_id_cache_write: false (default) reuses instance_id_cache_file JSON.
# Optional filtered QA dump: write_filtered_annotations_to in dataset_init (rank 0 only).
#
# Single GPU (default):
#   REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d.pth bash scripts/run_surprise_finetune.sh
#
# Multi-GPU on one node (torchrun); Slurm must allocate the same GPU count, e.g. --gres=gpu:4:
#   NPROC=4 REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d.pth bash scripts/run_surprise_finetune.sh
# (NPROC>1 forces run.distributed=true; effective batch ≈ batch_size_train * NPROC * accum_grad_iters.)
#
# Optional runner resume (training state from output_dir):
#   REASON3D_RESUME_CKPT=/path/to/ckpt_epoch_5.pth ...
# Intra-epoch checkpoints (e.g. every 2000 optimizer steps) + resume from mid-epoch:
#   REASON3D_TRAIN_OPTIONS="run.save_every_n_steps=2000" ...
#   REASON3D_RESUME_CKPT=/path/to/.../checkpoint_ep0003_iter00002000.pth ...
#
# Extra config overrides (space-separated key=value):
#   REASON3D_TRAIN_OPTIONS="run.max_epoch=50 run.output_dir=output/my_run" ...
set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"
export PYTHONPATH="${REASON3D}:${PYTHONPATH:-}"

CFG="${CFG:-lavis/projects/reason3d/train/reason3d_surprise_finetune_v2.yaml}"
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
OPTS+=( "datasets.3d_refer.dataset_init.pth_rel_subdir=${REASON3D_PTH_SUBDIR}" )
if [[ -n "${REASON3D_PTS_ROOT:-}" ]]; then
  OPTS+=( "datasets.3d_refer.build_info.points.storage=${REASON3D_PTS_ROOT}" )
fi
if [[ -n "${REASON3D_INSTANCE_CACHE_FILE:-}" ]]; then
  OPTS+=( "datasets.3d_refer.dataset_init.instance_id_cache_file=${REASON3D_INSTANCE_CACHE_FILE}" )
fi
if [[ "${REASON3D_FILTER_MISSING_GT_IN_PTH:-1}" == "0" ]]; then
  OPTS+=( "datasets.3d_refer.dataset_init.filter_missing_gt_in_pth=false" )
else
  OPTS+=( "datasets.3d_refer.dataset_init.filter_missing_gt_in_pth=true" )
fi

if [[ -n "${REASON3D_TRAIN_OPTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  OPTS+=( ${REASON3D_TRAIN_OPTIONS} )
fi

NPROC="${NPROC:-1}"
# reason3d_surprise_finetune.yaml sets run.distributed: false for single-GPU; torchrun alone is not enough.
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
