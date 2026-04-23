#!/usr/bin/env bash
# Fetch Google sparsehash (header-only) for pointgroup_ops when conda cannot install
# `conda-forge::sparsehash` (e.g. env stuck on InvalidSpec for nvidia::cuda-compiler).
# Uses the upstream 2.0.4 tag; `#include <google/dense_hash_map>` needs
#   POINTGROUP_SPARSEHASH_ROOT=.../src
#
# The GitHub tarball does *not* ship `src/sparsehash/internal/sparseconfig.h`; autotools
# generate it from `src/config.h` (via `./configure` + `make .../sparseconfig.h`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REASON3D_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${REASON3D_ROOT}/third_party/sparsehash-upstream-2.0.4"
INNER="sparsehash-sparsehash-2.0.4"
TAG="sparsehash-2.0.4"
URL="https://github.com/sparsehash/sparsehash/archive/refs/tags/${TAG}.tar.gz"

gen_sparseconfig() {
  # Requires: sh, ./configure, make, gawk (Makefile rule for sparseconfig.h).
  ( cd "$OUT" && ./configure && make src/sparsehash/internal/sparseconfig.h )
}

if [[ -f "$OUT/src/sparsehash/internal/sparseconfig.h" && -f "$OUT/src/google/dense_hash_map" ]]; then
  echo "Already present: $OUT"
  echo
  echo "  export POINTGROUP_SPARSEHASH_ROOT=\"$OUT/src\""
  exit 0
fi

# Half-finished tree from an older prep (extract only, no configure): finish in place.
if [[ -d "$OUT" && -f "$OUT/configure" && -f "$OUT/src/google/dense_hash_map" && ! -f "$OUT/src/sparsehash/internal/sparseconfig.h" ]]; then
  echo "Completing sparsehash (generating sparseconfig.h) in: $OUT"
  gen_sparseconfig
  echo
  echo "  export POINTGROUP_SPARSEHASH_ROOT=\"$OUT/src\""
  exit 0
fi

# Avoid mktemp(1) with a broken/missing TMPDIR (common on HPC: newline in env, or NFS path not created).
TDIR="${REASON3D_ROOT}/.tmp_prep_sparsehash_$$"
mkdir -p "$TDIR"
trap 'rm -rf "$TDIR"' EXIT
echo "Downloading $URL ..."
curl -fL -o "$TDIR/src.tgz" "$URL"
tar -xzf "$TDIR/src.tgz" -C "$TDIR"
mkdir -p "${REASON3D_ROOT}/third_party"
rm -rf "$OUT"
mv "$TDIR/${INNER}" "$OUT"
echo "Configuring and generating sparseconfig.h (needs gawk, a C/C++ compiler) ..."
gen_sparseconfig
echo "Extracted and prepared: $OUT"
echo
echo "For this shell only:"
echo "  export POINTGROUP_SPARSEHASH_ROOT=\"$OUT/src\""
echo
echo "Or run: bash scripts/build_pointgroup_ops.sh  (it auto-sets this if the path exists and POINTGROUP_SPARSEHASH_ROOT is unset)."
