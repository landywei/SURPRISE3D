# Data and artifacts sync (new GPU cluster)

Paths below match the **default YAMLs** in this fork (`/data/...`). Adjust source/destination if your old cluster used different mounts.

## 1. What the code expects on the new machine

| Role | Typical path | Notes |
|------|----------------|------|
| ScanNet++ root + prepared `.pth` | `/data/scannetpp` | Includes `processed/` (or `processed_trial/`) with `<scene_id>.pth` used by `pth_rel_subdir` |
| SPFormer backbone | `/data/checkpoints/spf_scannet_512.pth` | `point_encoder_cfg.pretrained` |
| Full Reason3D weights | your choice | Passed as `REASON3D_CKPT` / `REASON3D_INIT_CKPT` → `model.reason3d_checkpoint` |
| Surprise annotations | `/data/annotations/surprise_train.json`, `surprise_val.json` | YAML `build_info.annotations` |
| Optional: ScanNet++ tools / maps | under `scannetpp` | e.g. `benchmark_file_lists`, `data/` meshes — only if you re-run preprocessing |

**Hugging Face model cache** (not under `/data` by default): first run downloads **FLAN-T5-XL**, **BERT**, etc. Either:

- Rsync `~/.cache/huggingface/` (or your `HF_HOME`) from the old cluster, or  
- Let the new cluster download once and keep a **shared** `HF_HOME` on fast storage.

## 2. Large directories to rsync

Run from a machine that can **SSH** to both hosts (replace `OLD`, `NEW`, user, paths).

### ScanNet++ tree (large)

```bash
rsync -avh --progress OLD:/data/scannetpp/ NEW:/data/scannetpp/
```

If you only need **processed** point clouds for Reason3D:

```bash
rsync -avh --progress OLD:/data/scannetpp/processed/ NEW:/data/scannetpp/processed/
```

Add other subdirs (`data/`, `semantic/`, …) only if preprocessing or `update_superpoints.py` needs them.

### Checkpoints (small–medium)

```bash
rsync -avh --progress OLD:/data/checkpoints/ NEW:/data/checkpoints/
```

Include at least:

- `spf_scannet_512.pth`  
- Your **Reason3D** `.pth` used for eval/finetune.

### Annotations

```bash
rsync -avh --progress OLD:/data/annotations/ NEW:/data/annotations/
```

Minimum: `surprise_train.json`, `surprise_val.json` (and any trial JSONs you use).

## 3. Evaluation logs and qualitative outputs

Training/eval writes under the **gitignored** tree (do **not** commit these; sync separately):

```text
Models/reason3d/lavis/output/<run_cfg.output_dir>/<job_id>/
```

Examples:

- `.../reason3d_surprise_zeroshot_small/<timestamp>/qualitative/predictions.jsonl`
- `.../reason3d_surprise_zeroshot_small/<timestamp>/qualitative/masks/*.npz`
- `.../reason3d_surprise_finetune/<timestamp>/log.txt` (when `log_config` / training logs exist)

**Sync example** (old repo path → new):

```bash
rsync -avh --progress \
  OLD:/path/to/spatial_reasoning/Models/reason3d/lavis/output/ \
      NEW:/path/to/spatial_reasoning/Models/reason3d/lavis/output/
```

Or only one run:

```bash
rsync -avh --progress \
  OLD:.../lavis/output/reason3d_surprise_zeroshot_small/20260422191/ \
  NEW:.../lavis/output/reason3d_surprise_zeroshot_small/20260422191/
```

## 4. Optional: `processed_trial` or custom dirs

If you use `pth_rel_subdir: processed_trial` or custom annotation paths, sync those directories the same way and set YAML or `--options` on the new cluster.

## 5. Verification on the new cluster

```bash
# Spot-check counts
ls /data/scannetpp/processed/*.pth | wc -l
ls -la /data/checkpoints/
ls -la /data/annotations/surprise*.json
```

Then run `scripts/run_surprise_zeroshot_eval_small.sh` with `REASON3D_CKPT` pointing at the synced Reason3D checkpoint.
