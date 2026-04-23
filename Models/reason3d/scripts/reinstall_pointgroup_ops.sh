#!/usr/bin/env bash
# Clean a broken pointgroup_ops editable install and rebuild.
# Use when pip left half-installed state or CUDA env vars pointed at a missing toolkit.
set -euo pipefail

# shellcheck source=/dev/null
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/conda_init_reason3d.sh"

echo "Uninstalling pointgroup_ops (ignore 'not installed' warnings)"
pip uninstall -y pointgroup_ops 2>/dev/null || true

# Optional: refresh NVIDIA CUDA 12.1.1 compiler + headers in conda (matches PyTorch cu121 / install_reason3d_deps.sh):
# conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/build_pointgroup_ops.sh"
