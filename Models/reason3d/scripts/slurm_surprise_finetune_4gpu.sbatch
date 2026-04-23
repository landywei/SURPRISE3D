#!/bin/bash
# Submit from anywhere:
#   sbatch /nfs-stor/lan.wei/SURPRISE3D/Models/reason3d/scripts/slurm_surprise_finetune_4gpu.sbatch
#
# GPU partition (CIAI / CSCC): max 12h per job, up to 8 GPUs with cscc-gpu-qos.
# Edit #SBATCH lines if you use fewer GPUs (match --gres and NPROC).

#SBATCH -J reason3d-ft-4g
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH -o /nfs-stor/lan.wei/SURPRISE3D/Models/reason3d/lavis/output/slurm-%x-%j.out
#SBATCH -e /nfs-stor/lan.wei/SURPRISE3D/Models/reason3d/lavis/output/slurm-%x-%j.err

set -euo pipefail

REASON3D="/nfs-stor/lan.wei/SURPRISE3D/Models/reason3d"
cd "$REASON3D"

export HF_HOME="${HF_HOME:-/nfs-stor/lan.wei/data/hf_cache}"

# Non-login batch jobs: same conda bootstrap as run_surprise_finetune.sh uses internally.
# shellcheck source=/dev/null
. "${REASON3D}/scripts/conda_init_reason3d.sh"

REASON3D_TRAIN_OPTIONS="run.batch_size_train=1" \
NPROC=4 \
REASON3D_INIT_CKPT="/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth" \
  bash scripts/run_surprise_finetune.sh
