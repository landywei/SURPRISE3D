#!/usr/bin/env python3
"""Build map_benchmark_with_counts.csv (adds `count`) from official map_benchmark.csv."""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="/data/scannetpp/metadata/semantic_benchmark/map_benchmark.csv",
        help="Official ScanNet++ map_benchmark.csv",
    )
    p.add_argument(
        "--dst",
        default="/home/ubuntu/scannetpp_tools/semantic/configs/map_benchmark_with_counts.csv",
        help="Output path for prepare_training_data mapping_file",
    )
    args = p.parse_args()

    df = pd.read_csv(args.src)
    if "count" not in df.columns:
        df = df.copy()
        df["count"] = 1
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    df.to_csv(args.dst, index=False)
    print("Wrote", args.dst, "rows=", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
