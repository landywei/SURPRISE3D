#!/usr/bin/env bash
# Build the pointgroup_ops CUDA extension (required for import lavis... / evaluate.py).
#
# Typical failures:
#   - cuda_runtime_api.h: PyTorch guessed CUDA_HOME=$CONDA_PREFIX but headers live under
#     $CONDA_PREFIX/targets/x86_64-linux/include → we set CUDA_INC_PATH + CPATH.
#   - CUDA 12.x vs PyTorch cu118 (11.8): unset stale CUDA_HOME so conda's nvcc 11.8 is used,
#     or install CUDA 11.8 toolkit and set REASON3D_CUDA_HOME=/usr/local/cuda-11.8
#
# Optional: REASON3D_CUDA_HOME=/usr/local/cuda-11.8 (must match torch.version.cuda major)
set -euo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lavis/models/reason3d_models/lib" && pwd)"
echo "Building in: $LIB"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y libsparsehash-dev
fi

eval "$(conda shell.bash hook)"
conda activate reason3d310 2>/dev/null || conda activate reason3d 2>/dev/null || true

# Drop broken CUDA roots (empty /usr/local/cuda-11.8 tree, old profile exports, etc.)
if [[ -n "${REASON3D_CUDA_HOME:-}" && ! -x "${REASON3D_CUDA_HOME}/bin/nvcc" ]]; then
  echo "WARN: REASON3D_CUDA_HOME=${REASON3D_CUDA_HOME} has no executable bin/nvcc — unsetting." >&2
  unset REASON3D_CUDA_HOME
fi
if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "WARN: CUDA_HOME=${CUDA_HOME} has no executable bin/nvcc — unsetting CUDA_HOME/CUDA_PATH." >&2
  unset CUDA_HOME CUDA_PATH
fi

torch_cuda_maj() {
  python - <<'PY'
import re
import torch
v = torch.version.cuda or ""
m = re.match(r"(\d+)", v)
print(m.group(1) if m else "0")
PY
}

nvcc_maj() {
  local o
  o="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
  echo "${o:-0}"
}

# Prefer a side-by-side CUDA 11.8 toolkit (matches typical PyTorch cu118 wheels).
if [[ -z "${REASON3D_CUDA_HOME:-}" && -x /usr/local/cuda-11.8/bin/nvcc ]]; then
  export REASON3D_CUDA_HOME="/usr/local/cuda-11.8"
  echo "Auto-selected REASON3D_CUDA_HOME=$REASON3D_CUDA_HOME"
fi

# User override: full NVIDIA toolkit matching PyTorch CUDA (recommended for reliable builds).
if [[ -n "${REASON3D_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="${REASON3D_CUDA_HOME}"
  export CUDA_PATH="${REASON3D_CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "Using REASON3D_CUDA_HOME=$CUDA_HOME"
else
  # Prefer conda's nvcc on PATH; drop CUDA_HOME/CUDA_PATH so PyTorch infers from `which nvcc`
  # (avoids CUDA_HOME=/usr/local/cuda 12.x with PyTorch 11.8).
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    unset CUDA_HOME CUDA_PATH || true
  fi
fi

TM="$(torch_cuda_maj)"
NM="$(nvcc_maj)"
if [[ "$TM" != "0" && "$NM" != "0" && "$TM" != "$NM" ]]; then
  echo "ERROR: nvcc major CUDA ($NM) != PyTorch torch.version.cuda major ($TM)." >&2
  echo "  Fix: install CUDA toolkit $TM.x and export REASON3D_CUDA_HOME=/path/to/cuda-$TM" >&2
  echo "  or install a PyTorch build whose CUDA major matches your nvcc." >&2
  exit 1
fi

# Conda nvidia packages: headers under targets/..., not $CONDA_PREFIX/include
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  TGT_INC="$CONDA_PREFIX/targets/x86_64-linux/include"
  TGT_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
  if [[ -f "$TGT_INC/cuda_runtime_api.h" ]]; then
    export CUDA_INC_PATH="${TGT_INC}"
    export CPATH="${TGT_INC}${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${TGT_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${TGT_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Conda CUDA includes: $TGT_INC"
    echo "Note: if the build then fails on thrust/complex.h, install NVIDIA CUDA Toolkit 11.8" >&2
    echo "  (matching PyTorch cu118) and run with REASON3D_CUDA_HOME=/usr/local/cuda-11.8" >&2
  fi
fi

cd "$LIB"
pip install -e . --no-build-isolation

python -c "import pointgroup_ops; print('pointgroup_ops OK')"
