#!/usr/bin/env bash
# Deprecated: use reinstall_pytorch_stack_cu121.sh (PyTorch cu121 / CUDA 12.x).
set -euo pipefail
echo "NOTE: reinstall_pytorch_stack_cu118.sh is deprecated; forwarding to reinstall_pytorch_stack_cu121.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reinstall_pytorch_stack_cu121.sh" "$@"
