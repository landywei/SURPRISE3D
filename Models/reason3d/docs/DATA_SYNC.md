# Data and artifacts sync (new GPU cluster)

Paths below match the **default YAMLs** in this fork (`/nfs-stor/lan.wei/data/...`). Adjust source/destination if your hosts use different mounts.

## 1. What the code expects on the new machine

| Role | Typical path | Notes |
|------|----------------|------|
| ScanNet++ root + prepared `.pth` | `/nfs-stor/lan.wei/data/scannetpp` | Includes `processed/` (or `processed_trial/`) with `<scene_id>.pth` used by `pth_rel_subdir` |
| SPFormer backbone | `/nfs-stor/lan.wei/data/checkpoints/spf_scannet_512.pth` | `point_encoder_cfg.pretrained` |
| Full Reason3D weights | your choice | Passed as `REASON3D_CKPT` / `REASON3D_INIT_CKPT` → `model.reason3d_checkpoint` |
| Surprise annotations | `/nfs-stor/lan.wei/data/annotations/surprise_train.json`, `surprise_val.json` | YAML `build_info.annotations` |
| Optional: ScanNet++ tools / maps | under `scannetpp` | e.g. `benchmark_file_lists`, `data/` meshes — only if you re-run preprocessing |

### What **training** (`train.py` / `ThreeDReferDataset`) actually reads under `/nfs-stor/lan.wei/data/scannetpp`

With the default Surprise finetune config (`build_info.points.storage: /nfs-stor/lan.wei/data/scannetpp` and `dataset_init.pth_rel_subdir: processed`), the dataloader **only** touches:

```text
/nfs-stor/lan.wei/data/scannetpp/processed/<scene_id>.pth
```

one file per scene that appears in your train/val JSON (and, with `filter_missing_gt_in_pth`, each unique scene is loaded once at init to build the instance-id cache). Keys used from each `.pth` include at least: `sampled_coords`, `sampled_colors`, `superpoints`, `sampled_labels`, `sampled_instance_anno_id` (and related tensors the collater expects).

**You do not need** the rest of the ScanNet++ tree (`data/<scene>/meshes`, `semantic/`, raw scans, etc.) **for training/inference** if those `.pth` files are already built on the old cluster.

**Sync first (minimal):**

```bash
# Replace OLD:/<path>/ with the source host’s ScanNet++ root (e.g. OLD:/data/scannetpp/).
rsync -avh --progress OLD:<path-on-source>/scannetpp/processed/ NEW:/nfs-stor/lan.wei/data/scannetpp/processed/
```

If you use **`pth_rel_subdir: processed_trial`** (or another folder name), sync **that** directory instead.

**Separate from training:** rebuilding `.pth` or running `update_superpoints.py` needs extra paths (e.g. meshes under `data/<scene_id>/scans/mesh_aligned_0.05.ply` and UniDet `*superpoints.npy`). See `update_superpoints.py` and your `scannetpp_tools` prep pipeline—only rsync those if you will **re-preprocess** on the new cluster.

**Instance-id cache (optional, small JSON):** if you use `dataset_init.instance_id_cache_file` under `processed/` (see `reason3d_surprise_finetune.yaml`), **rsync that file too** so the new cluster skips the long “filter” `torch.load` pass on first job.

**Hugging Face model cache** (separate from the dataset paths above; usually under `~/.cache/huggingface` or `HF_HOME`): first run downloads **FLAN-T5-XL**, **BERT**, etc. Either:

- Rsync `~/.cache/huggingface/` (or your `HF_HOME`) from the old cluster, or  
- Let the new cluster download once and keep a **shared** `HF_HOME` on fast storage.

## 2. Large directories to rsync

Run from a machine that can **SSH** to both hosts (replace `OLD`, `NEW`, user, and each path). The **destination** in this repo defaults to `/nfs-stor/lan.wei/data/...`; the **source** may still be something like `/data/...` on an older machine.

### ScanNet++ tree (large)

```bash
rsync -avh --progress OLD:<path-on-source>/scannetpp/ NEW:/nfs-stor/lan.wei/data/scannetpp/
```

If you only need **processed** point clouds for Reason3D:

```bash
rsync -avh --progress OLD:<path-on-source>/scannetpp/processed/ NEW:/nfs-stor/lan.wei/data/scannetpp/processed/
```

Add other subdirs (`data/`, `semantic/`, …) only if preprocessing or `update_superpoints.py` needs them.

### Checkpoints (small–medium)

```bash
rsync -avh --progress OLD:<path-on-source>/checkpoints/ NEW:/nfs-stor/lan.wei/data/checkpoints/
```

Include at least:

- `spf_scannet_512.pth`  
- Your **Reason3D** `.pth` used for eval/finetune.

### Annotations

```bash
rsync -avh --progress OLD:<path-on-source>/annotations/ NEW:/nfs-stor/lan.wei/data/annotations/
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
ls /nfs-stor/lan.wei/data/scannetpp/processed/*.pth | wc -l
ls -la /nfs-stor/lan.wei/data/checkpoints/
ls -la /nfs-stor/lan.wei/data/annotations/surprise*.json
```

Then run `scripts/run_surprise_zeroshot_eval_small.sh` with `REASON3D_CKPT` pointing at the synced Reason3D checkpoint.
