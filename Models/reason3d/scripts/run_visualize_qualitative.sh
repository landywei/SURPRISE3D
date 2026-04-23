#!/usr/bin/env bash
# Wrapper for scripts/visualize_qualitative_preds.py with sensible defaults.
#
# Usage:
#   QUAL_DIR=lavis/output/reason3d_surprise_zeroshot_small/20260423154/qualitative \
#   REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp \
#   bash scripts/run_visualize_qualitative.sh --list
#
#   bash scripts/run_visualize_qualitative.sh --export-row 0 --out-dir /tmp/qvis
#
# Env:
#   QUAL_DIR          - qualitative folder (required unless first arg looks like a path)
#   REASON3D_PTS_ROOT - points root (default from small zeroshot YAML: scannetpp tree)
#   REASON3D_PTH_SUBDIR - default processed

set -euo pipefail

REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REASON3D"

QUAL_DIR="${QUAL_DIR:-}"
if [[ -n "${1:-}" && "${1:0:1}" != "-" ]]; then
  QUAL_DIR="$1"
  shift
fi

if [[ -z "$QUAL_DIR" ]]; then
  echo "Set QUAL_DIR to .../qualitative or pass it as the first argument." >&2
  exit 1
fi

export REASON3D_PTS_ROOT="${REASON3D_PTS_ROOT:-/nfs-stor/lan.wei/data/scannetpp}"
export REASON3D_PTH_SUBDIR="${REASON3D_PTH_SUBDIR:-processed}"

exec python scripts/visualize_qualitative_preds.py --qual-dir "$QUAL_DIR" "$@"
