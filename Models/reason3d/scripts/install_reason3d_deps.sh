#!/usr/bin/env bash
# First-pass install: CUDA 12.1.1 toolkit in conda (nvcc + headers) + Python deps from requirements.txt
# (PyTorch 2.1.2+cu121 + PyPI). Requires network. Activate your conda env first (e.g. reason3d310).
#
# Version alignment (matches requirements.txt / cu121 wheels):
#   - PyTorch cu121 → CUDA runtime 12.1.x in the wheel
#   - Conda: NVIDIA channel label cuda-12.1.1 → package cuda-toolkit (bundles cuda-nvcc, cuda-cudart-dev,
#     libraries and CCCL headers so nvcc major matches torch.version.cuda major 12)
#
# Does not build pointgroup_ops (run after this): bash scripts/build_pointgroup_ops.sh
#
# Conda in non-login / srun shells: scripts/conda_init_reason3d.sh; override with
#   REASON3D_CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
# Optional:
#   REASON3D_PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 (default)
#   REASON3D_CUDA_CONDA_LABEL=nvidia/label/cuda-12.1.1 (default; use cuda-12.1.0 for the prior 12.1 bundle)
#   REASON3D_SKIP_CONDA_CUDA=1 — skip conda CUDA (use system /usr/local/cuda-12.* or broken conda env)
#   REASON3D_SKIP_CONDA_BUILD_TOOLS=1 — skip ninja + gxx_linux-64 (host already has a good g++ / ninja)
#   REASON3D_SKIP_TORCH_SCATTER=1 — skip torch-scatter (CPU-only / unusual layout only)
#   REASON3D_PYG_WHEEL_INDEX — PyG wheel page (default matches torch 2.1.2+cu121)
#   REASON3D_SKIP_SPARSEHASH=1 — skip conda-forge::sparsehash (use prep_google_sparsehash_headers.sh instead)
set -euo pipefail

# conda activate/deactivate hooks (e.g. gcc_linux-64) reference vars that may be unset; bash -u breaks them.
_reason3d_conda() {
  local _restore=
  if [[ $- == *u* ]]; then
    set +u
    _restore=1
  fi
  "$@"
  local _st=$?
  if [[ -n "${_restore:-}" ]]; then
    set -u
  fi
  return "$_st"
}

REASON3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D_ROOT"

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${_LIB_DIR}/conda_init_reason3d.sh"
unset _LIB_DIR

PYTORCH_INDEX="${REASON3D_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# Pinned NVIDIA label so all CUDA components resolve to one 12.1.1 stack (avoids mixed 12.x metapackages).
CUDA_CONDA_LABEL="${REASON3D_CUDA_CONDA_LABEL:-nvidia/label/cuda-12.1.1}"

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not on PATH after conda init. Activate an env or set REASON3D_CONDA_SH." >&2
  exit 1
fi

if [[ "${REASON3D_SKIP_CONDA_CUDA:-0}" == "1" ]]; then
  echo "SKIP: REASON3D_SKIP_CONDA_CUDA=1 — not installing conda cuda-toolkit (install nvcc 12.1 yourself)."
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "WARN: conda not on PATH; skipping CUDA toolkit install. Use system CUDA 12.1+ or install conda." >&2
  elif [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "WARN: CONDA_PREFIX unset; skipping conda CUDA install (activate a conda env first)." >&2
  else
    echo "Installing CUDA toolkit 12.1.1 (nvcc + dev libraries) from channel: $CUDA_CONDA_LABEL"
    echo "  into conda prefix: $CONDA_PREFIX"
    _reason3d_conda conda install -y -c "${CUDA_CONDA_LABEL}" cuda-toolkit
    if command -v nvcc >/dev/null 2>&1; then
      nvcc -V || true
    fi
  fi
fi

# Ninja + a known-good host GCC for nvcc/pybind11 extension builds (pointgroup_ops).
if [[ "${REASON3D_SKIP_CONDA_BUILD_TOOLS:-0}" != "1" ]] && command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Installing conda build tools: ninja, gxx_linux-64=11.2.0"
  _reason3d_conda conda install -y -c conda-forge ninja
  if ! _reason3d_conda conda install -y -c defaults "gxx_linux-64=11.2.0"; then
    echo "WARN: could not install gxx_linux-64=11.2.0 from defaults; ensure host has g++ (e.g. gcc-toolset) for nvcc." >&2
  fi
fi

# Google sparsehash headers for pointgroup_ops (`#include <google/dense_hash_map>`). Not on PyPI as a pip dep.
if [[ "${REASON3D_SKIP_SPARSEHASH:-0}" != "1" ]] && command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Installing conda-forge::sparsehash (C++ headers into conda prefix)"
  if ! _reason3d_conda conda install -y -c conda-forge sparsehash; then
    echo "WARN: conda sparsehash failed (solver/InvalidSpec). Run on a login node: bash scripts/prep_google_sparsehash_headers.sh" >&2
  fi
fi

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  --extra-index-url "$PYTORCH_INDEX" \
  -r "${REASON3D_ROOT}/requirements.txt"

# torch-scatter: prebuilt wheels from PyG (do not use torch-sparse/torch-cluster here — not imported by this repo).
PYG_INDEX="${REASON3D_PYG_WHEEL_INDEX:-https://data.pyg.org/whl/torch-2.1.2+cu121.html}"
if [[ "${REASON3D_SKIP_TORCH_SCATTER:-0}" == "1" ]]; then
  echo "SKIP: REASON3D_SKIP_TORCH_SCATTER=1 — not installing torch-scatter."
else
  echo "Installing torch-scatter from PyG wheel index: $PYG_INDEX"
  python -m pip install torch-scatter -f "$PYG_INDEX"
fi

echo ""
echo "Done. PyTorch extra-index: $PYTORCH_INDEX"
if [[ "${REASON3D_SKIP_CONDA_CUDA:-0}" == "1" ]]; then
  echo "Conda CUDA toolkit: skipped (REASON3D_SKIP_CONDA_CUDA=1)."
else
  echo "Conda CUDA toolkit: cuda-toolkit from $CUDA_CONDA_LABEL (CUDA 12.1.1 stack)."
fi
echo "Next: cd \"$REASON3D_ROOT\" && bash scripts/build_pointgroup_ops.sh"
echo "Smoke: cd \"$REASON3D_ROOT\" && bash scripts/smoke_env.sh"
