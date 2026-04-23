# shellcheck shell=bash
# Source from bash scripts: source "$(dirname ...)/conda_init_reason3d.sh"
# srun/Slurm shells often lack `conda` on PATH. Set REASON3D_CONDA_SH to your conda.sh if needed.
#
# Parent scripts often use `set -u`. Conda hooks and nvidia cuda-nvcc activate/deactivate
# scripts reference optional vars (e.g. CUDAARCHS_BACKUP); temporarily allow unset vars.
if [[ $- == *u* ]]; then
  set +u
  _REASON3D_RESTORE_NOUNSET=1
else
  _REASON3D_RESTORE_NOUNSET=
fi

if ! command -v conda >/dev/null 2>&1; then
  for f in \
    "${REASON3D_CONDA_SH:-}" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "$f" && -f "$f" ]]; then
      # shellcheck source=/dev/null
      . "$f"
      break
    fi
  done
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate reason3d310 2>/dev/null || conda activate reason3d 2>/dev/null || true
else
  if command -v python >/dev/null 2>&1; then
    PFX="$(python -c "import os, sys; p=os.path.realpath(sys.executable); print(os.path.dirname(os.path.dirname(p)))" 2>/dev/null || true)"
    if [[ -n "$PFX" && -d "$PFX" ]]; then
      export CONDA_PREFIX="${CONDA_PREFIX:-$PFX}"
      export PATH="${CONDA_PREFIX}/bin:${PATH}"
      echo "WARN: conda not on PATH; using python env CONDA_PREFIX=$CONDA_PREFIX" >&2
    fi
  fi
fi

if [[ -n "${_REASON3D_RESTORE_NOUNSET:-}" ]]; then
  set -u
  unset _REASON3D_RESTORE_NOUNSET
fi
