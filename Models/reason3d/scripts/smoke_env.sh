#!/usr/bin/env bash
# Quick import check after installing deps and (optionally) building pointgroup_ops.
# Run from login or GPU node: bash scripts/smoke_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi
exec "$PY" - <<'PY'
import sys
import numpy as np
import torch

print("python", sys.version.split()[0])
print("numpy", np.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
import transformers
print("transformers", transformers.__version__)
import lavis
print("lavis import: ok")
try:
    import pointgroup_ops
    print("pointgroup_ops: ok")
except Exception as exc:
    print("pointgroup_ops: FAILED —", exc)
    print("  (rebuild: bash scripts/build_pointgroup_ops.sh)")
    sys.exit(0)
PY
