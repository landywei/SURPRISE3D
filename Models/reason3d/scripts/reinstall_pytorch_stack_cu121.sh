#!/usr/bin/env bash
# Reinstall the PyTorch stack pinned in requirements.txt from the cu121 (CUDA 12.1) wheel index.
# Use when site-packages has mismatched torch/torchvision/torchaudio, for example:
#   ModuleNotFoundError: No module named 'triton.backends' (new torch + old triton)
#   ImportError: cannot import name 'ExportOptions' from 'torch.onnx._internal.exporter'
#     (new torchvision expecting newer torch.onnx than torch 2.1.x provides)
# Also reapplies numpy<2 (requirements.txt): PyTorch 2.1.x + NumPy 2.x causes _ARRAY_API / ABI warnings.
# Realign requests/charset-normalizer (avoids _FREQUENCIES_SET ImportError from mixed charset_normalizer files).
set -euo pipefail

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${_LIB_DIR}/conda_init_reason3d.sh"
unset _LIB_DIR

pip install --upgrade --force-reinstall \
  "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2" "triton==2.1.0" \
  --index-url "https://download.pytorch.org/whl/cu121"

pip install "numpy>=1.21,<2"

pip install --force-reinstall "charset-normalizer>=3.2,<4" "requests>=2.28,<3"

PYG_INDEX="${REASON3D_PYG_WHEEL_INDEX:-https://data.pyg.org/whl/torch-2.1.2+cu121.html}"
pip install --force-reinstall torch-scatter -f "$PYG_INDEX"

pip install --force-reinstall "spconv-cu121>=2.3.0"
