#!/usr/bin/env bash
# Build the pointgroup_ops CUDA extension (required for import lavis... / evaluate.py).
#
# Typical failures:
#   - cuda_runtime_api.h: PyTorch guessed CUDA_HOME=$CONDA_PREFIX but headers live under
#     $CONDA_PREFIX/targets/x86_64-linux/include → we set CUDA_INC_PATH + CPATH.
#   - Wrong CUDA toolkit on PATH vs PyTorch (cu121): unset stale CUDA_HOME or set
#     REASON3D_CUDA_HOME to a full CUDA 12.x install whose nvcc major matches torch.version.cuda.
#
# Optional: REASON3D_CUDA_HOME=/usr/local/cuda-12.6 (must match torch.version.cuda major, e.g. 12)
# Conda in non-login / srun shells: scripts/conda_init_reason3d.sh sources conda.sh; override with
#   REASON3D_CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
# Optional: REASON_PYTHON=/path/to/python with torch (default: $CONDA_PREFIX/bin/python when present)
set -euo pipefail

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${_LIB_DIR}/conda_init_reason3d.sh"
unset _LIB_DIR

# Prefer env Python (has torch); bare `python` may be system/miniconda base without torch.
if [[ -n "${REASON_PYTHON:-}" && -x "${REASON_PYTHON}" ]]; then
  REASON_PY="${REASON_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  REASON_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  REASON_PY=python3
else
  REASON_PY=python
fi
if ! "$REASON_PY" -c "import torch" 2>/dev/null; then
  echo "ERROR: PyTorch not found in: $REASON_PY" >&2
  echo "  Activate the conda env that has torch (e.g. conda activate reason3d310), then re-run." >&2
  exit 1
fi

REASON3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lavis/models/reason3d_models/lib" && pwd)"
echo "Building in: $LIB"

# Vendored headers from scripts/prep_google_sparsehash_headers.sh (avoids conda when env is broken).
_VEND_SH="${REASON3D_ROOT}/third_party/sparsehash-upstream-2.0.4/src"
if [[ -f "${_VEND_SH}/google/dense_hash_map" && -z "${POINTGROUP_SPARSEHASH_ROOT:-}" ]]; then
  export POINTGROUP_SPARSEHASH_ROOT="${_VEND_SH}"
  echo "Using Google sparsehash headers: POINTGROUP_SPARSEHASH_ROOT=$POINTGROUP_SPARSEHASH_ROOT"
fi
unset _VEND_SH

# Drop broken CUDA roots (empty /usr/local/cuda-12.* tree, old profile exports, etc.)
if [[ -n "${REASON3D_CUDA_HOME:-}" && ! -x "${REASON3D_CUDA_HOME}/bin/nvcc" ]]; then
  echo "WARN: REASON3D_CUDA_HOME=${REASON3D_CUDA_HOME} has no executable bin/nvcc — unsetting." >&2
  unset REASON3D_CUDA_HOME
fi
if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "WARN: CUDA_HOME=${CUDA_HOME} has no executable bin/nvcc — unsetting CUDA_HOME/CUDA_PATH." >&2
  unset CUDA_HOME CUDA_PATH
fi

torch_cuda_maj() {
  "$REASON_PY" - <<'PY'
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

# Prefer a side-by-side CUDA 12.x toolkit (matches PyTorch cu121 wheels; pick newest present).
if [[ -z "${REASON3D_CUDA_HOME:-}" ]]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda-12.4 /usr/local/cuda-12.3 \
               /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -x "${_cand}/bin/nvcc" ]]; then
      export REASON3D_CUDA_HOME="$_cand"
      echo "Auto-selected REASON3D_CUDA_HOME=$REASON3D_CUDA_HOME"
      break
    fi
  done
  unset _cand
fi

# User override: full NVIDIA toolkit matching PyTorch CUDA (recommended for reliable builds).
if [[ -n "${REASON3D_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="${REASON3D_CUDA_HOME}"
  export CUDA_PATH="${REASON3D_CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "Using REASON3D_CUDA_HOME=$CUDA_HOME"
else
  # Prefer conda's nvcc on PATH; drop CUDA_HOME/CUDA_PATH so PyTorch infers from `which nvcc`
  # (avoids CUDA_HOME pointing at a toolkit whose major mismatches torch.version.cuda).
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
    echo "Note: if the build then fails on thrust/complex.h, install a full NVIDIA CUDA 12.x toolkit" >&2
    echo "  matching torch.version.cuda and run with REASON3D_CUDA_HOME=/usr/local/cuda-12.x" >&2
  fi
fi

cd "$LIB"
"$REASON_PY" -m pip install -e . --no-build-isolation

"$REASON_PY" -c "import pointgroup_ops; print('pointgroup_ops OK')"
