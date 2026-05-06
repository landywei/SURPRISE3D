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
#   QUAL_DIR             - qualitative folder (required unless first arg looks like a path)
#   REASON3D_PTS_ROOT    - points root (default from small zeroshot YAML: scannetpp tree)
#   REASON3D_PTH_SUBDIR  - default processed
#   WITH_GT              - set to 1 to also write *_gt.ply (default: skip GT overlay)
#   WITH_RGB             - set to 1 to also write *_rgb.ply (default: skip RGB cloud)
#   SKIP_INTERMEDIATE    - set to 1 to suppress *_pred_intermediate.ply on chainv3-CoT runs
#                          (default: write it whenever the .npz carries
#                          pred_pmask_intermediate)
#   INTERMEDIATE_ONLY    - set to 1 to skip *_pred.ply and only render the M_1 PLY

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

# Default: skip both GT and RGB overlays (only *_pred.ply is written). Either
# can be re-enabled with WITH_GT=1 / WITH_RGB=1 or by passing --with-gt /
# --with-rgb (or their --skip-* counterparts) on the command line. We only
# inject our defaults when the caller has not already chosen a side.
extra_args=()

user_set_gt=0
for arg in "$@"; do
  case "$arg" in
    --skip-gt|--with-gt) user_set_gt=1; break ;;
  esac
done
if [[ "$user_set_gt" -eq 0 ]]; then
  if [[ "${WITH_GT:-0}" == "1" ]]; then
    extra_args+=(--with-gt)
  else
    extra_args+=(--skip-gt)
  fi
fi

user_set_rgb=0
for arg in "$@"; do
  case "$arg" in
    --skip-rgb|--with-rgb) user_set_rgb=1; break ;;
  esac
done
if [[ "$user_set_rgb" -eq 0 ]]; then
  if [[ "${WITH_RGB:-0}" == "1" ]]; then
    extra_args+=(--with-rgb)
  else
    extra_args+=(--skip-rgb)
  fi
fi

# chainv3-CoT intermediate (M_1) overlay: emit by default whenever the .npz
# contains pred_pmask_intermediate; SKIP_INTERMEDIATE=1 disables, and an
# explicit --skip-intermediate / --with-intermediate / --intermediate-only on
# the command line wins over both.
user_set_inter=0
for arg in "$@"; do
  case "$arg" in
    --skip-intermediate|--with-intermediate|--intermediate-only) user_set_inter=1; break ;;
  esac
done
if [[ "$user_set_inter" -eq 0 ]]; then
  if [[ "${INTERMEDIATE_ONLY:-0}" == "1" ]]; then
    extra_args+=(--intermediate-only)
  elif [[ "${SKIP_INTERMEDIATE:-0}" == "1" ]]; then
    extra_args+=(--skip-intermediate)
  fi
fi

exec python scripts/visualize_qualitative_preds.py --qual-dir "$QUAL_DIR" ${extra_args[@]+"${extra_args[@]}"} "$@"
