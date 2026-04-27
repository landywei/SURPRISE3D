#!/usr/bin/env bash
# Copy ScanNet v2 scans into train/val, then build *_reason.pth via prepare_data_reason.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "Copy data"
python split_data.py

echo "Preprocess data (prepare_data_reason.py)"
python prepare_data_reason.py --data_split train
python prepare_data_reason.py --data_split val
