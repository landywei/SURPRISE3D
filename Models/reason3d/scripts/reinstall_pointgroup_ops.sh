#!/usr/bin/env bash
# Clean a broken pointgroup_ops editable install and rebuild.
# Use when pip left half-installed state or CUDA env vars pointed at a missing toolkit.
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate reason3d310 2>/dev/null || conda activate reason3d 2>/dev/null || true

echo "Uninstalling pointgroup_ops (ignore 'not installed' warnings)"
pip uninstall -y pointgroup_ops 2>/dev/null || true

# Optional: refresh NVIDIA CUDA 11.8 compiler + headers in conda (matches many cu118 wheels).
# Uncomment if you want conda to (re)install the toolkit pieces:
# conda install -y -c nvidia "cuda-toolkit=11.8" "cuda-nvcc=11.8"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/build_pointgroup_ops.sh"
