#!/usr/bin/env bash
# Trial: few scenes → processed_trial/, merge superpoints, slice JSON, verify.
set -euo pipefail

SCNNETPP="${SCNNETPP:-/data/scannetpp}"
TOOLS="${SCNNETPP_TOOLS:-/home/ubuntu/scannetpp_tools}"
REASON3D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$REASON3D/../.." && pwd)"
TRIAL_LIST="$REASON3D/scripts/trial_scenes.txt"
TRIAL_YML="$REASON3D/scripts/trial_prepare_training_data.yml"
UNIDET_SCENE_DIR="${SCNNETPP}/unidet3d_prep/data"
IDS_CSV="$(grep -v '^[[:space:]]*$' "$TRIAL_LIST" | paste -sd, - | tr -d '[:space:]')"

echo "=== Trial preprocess ==="
echo "SCNNETPP=$SCNNETPP TOOLS=$TOOLS REASON3D=$REASON3D"

mkdir -p "${SCNNETPP}/processed_trial"

echo "[0/5] mapping CSV with count column (required by ScanNet++ MapLabelToIndex)"
python3 "$REASON3D/scripts/ensure_map_benchmark_with_counts.py"

eval "$(conda shell.bash hook)"
conda activate reason3d310 2>/dev/null || conda activate reason3d 2>/dev/null || true

echo "[1/5] prepare_training_data → processed_trial/"
( cd "$TOOLS" && python -m semantic.prep.prepare_training_data "$TRIAL_YML" )

echo "[2/5] merge superpoints (reuse UniDet outputs under unidet3d_prep)"
( cd "$REASON3D" && python update_superpoints.py \
  --pth_dir "${SCNNETPP}/processed_trial" \
  --scene_dir "$UNIDET_SCENE_DIR" \
  --scannetpp_root "$SCNNETPP" \
  --only_scenes "$IDS_CSV" \
  --force )

echo "[3/5] trial annotations → /data/annotations/surprise_trial_train.json"
python3 <<PY
import json, os
scenes = {s.strip() for s in open("$TRIAL_LIST") if s.strip()}
with open("/data/annotations/surprise_train.json") as f:
    rows = json.load(f)
out = [r for r in rows if r.get("scene_id") in scenes]
os.makedirs("/data/annotations", exist_ok=True)
with open("/data/annotations/surprise_trial_train.json", "w") as f:
    json.dump(out, f)
print("wrote", len(out), "rows")
PY

echo "[4/5] verify (aggregate + per-scene)"
python3 "$REASON3D/scripts/verify_surprise3d_instance_ids.py" \
  --scannetpp-root "$SCNNETPP" \
  --train-json /data/annotations/surprise_trial_train.json \
  --pth-subdir processed_trial \
  --sample 10 --seed 0

while IFS= read -r s; do
  [[ -z "$s" ]] && continue
  echo "--- $s ---"
  python3 "$REASON3D/scripts/verify_surprise3d_instance_ids.py" \
    --scannetpp-root "$SCNNETPP" \
    --train-json /data/annotations/surprise_trial_train.json \
    --pth-subdir processed_trial \
    --scene "$s"
done < "$TRIAL_LIST"

echo "[5/5] Done."
