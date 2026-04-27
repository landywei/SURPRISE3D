# Finetune and eval scripts (inventory)

All paths below are relative to `Models/reason3d/` unless noted. Entry points: `train.py` (finetune) and `evaluate.py` (eval), run with `PYTHONPATH` including the repo root (the wrapper scripts set this). Conda is initialized via `scripts/conda_init_reason3d.sh` where the script sources it.

## PTH subdir (Surprise + ScanNet++)

Preprocessed point clouds are loaded from `join(points.storage, pth_rel_subdir, "<scene_id>.pth")`.

**Default** everywhere: `processed_surprise_full_pth` — set in `lavis/configs/datasets/3dseg/defaults.yaml`, `defaults_geo.yaml`, the Surprise finetune / zeroshot YAMLs, and the wrapper scripts as `REASON3D_PTH_SUBDIR` (so `--options` wins over the YAML in code). For smoke runs or older trees, set `REASON3D_PTH_SUBDIR=processed` or `processed_trial` (or pass `datasets.3d_refer*.dataset_init.pth_rel_subdir=...`).

---

## Quick reference

| Script | Role | Default config (`CFG`) | Python entry |
|--------|------|------------------------|--------------|
| `scripts/run_surprise_finetune.sh` | Finetune baseline `reason3d_t5` on Surprise + 3D refer | `lavis/projects/reason3d/train/reason3d_surprise_finetune.yaml` | `train.py` |
| `scripts/run_surprise_finetune_geo.sh` | Finetune `reason3d_t5_geo` + `3d_refer_geo` | `lavis/projects/reason3d/train/reason3d_surprise_finetune_geo.yaml` | `train.py` |
| `scripts/slurm_surprise_finetune_4gpu.sbatch` | Slurm: 4-GPU finetune via `run_surprise_finetune.sh` | (inherits script default) | `train.py` (via bash) |
| `scripts/run_surprise_zeroshot_eval.sh` | Full val eval, baseline model | `lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml` | `evaluate.py` |
| `scripts/run_surprise_zeroshot_eval_small.sh` | Small allowlist + caps (trial-style) | `lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small.yaml` | `evaluate.py` |
| `scripts/run_surprise_zeroshot_eval_small_geo.sh` | Geo model + `3d_refer_geo`, small eval | `lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small_geo.yaml` | `evaluate.py` |
| `scripts/run_visualize_qualitative.sh` | **Post-eval** PLY / viz for `qualitative/` outputs | N/A (wraps `visualize_qualitative_preds.py`) | Python script |

---

## Finetune

### `run_surprise_finetune.sh`

- **Model / data:** `3d_refer` dataset, standard Reason3D stack.
- **Required:** `REASON3D_INIT_CKPT` (or `REASON3D_CKPT`) — base Reason3D `.pth` with `checkpoint["model"]`.
- **Data env:** `REASON3D_PTH_SUBDIR` (default `processed_surprise_full_pth`, always passed); `REASON3D_PTS_ROOT`, `REASON3D_INSTANCE_CACHE_FILE` when set. **Optional:** `REASON3D_RESUME_CKPT` — training resume; `REASON3D_FILTER_MISSING_GT_IN_PTH=0` to keep rows without GT in `.pth`; `REASON3D_TRAIN_OPTIONS` (space-separated `key=value`); `NPROC` &gt; 1 adds `run.distributed=true` and uses `torchrun`.
- **Override config:** `CFG=.../other.yaml` (e.g. for experiments).
- **Notes:** Multi-GPU effective batch ≈ `batch_size_train × NPROC × accum_grad_iters`.

### `run_surprise_finetune_geo.sh`

- **Same interface** as `run_surprise_finetune.sh`, but options target **`datasets.3d_refer_geo.*`** (not `3d_refer`).
- **Default `CFG`:** `reason3d_surprise_finetune_geo.yaml` (`reason3d_t5_geo`, geometry-aware superpoint path).
- **Quick subset:** `CFG=lavis/projects/reason3d/train/reason3d_surprise_finetune_geo_quick.yaml` (smaller train cap / few epochs; **same** `geo_relational_cfg` as the full geo YAML).

### `slurm_surprise_finetune_4gpu.sbatch`

- **Hardcoded** `REASON3D` path, partition/QoS, 4 GPUs, 12h, log under `lavis/output/slurm-*.out/.err`.
- Invokes: `NPROC=4` + `REASON3D_TRAIN_OPTIONS="run.batch_size_train=1"` + `REASON3D_INIT_CKPT=.../reason3d_inference.pth` + `bash scripts/run_surprise_finetune.sh` (non-geo). **Edit the sbatch** if you need geo finetune or different init checkpoint.

---

## Eval (zero-shot / checkpoint)

### `run_surprise_zeroshot_eval.sh`

- **Required:** `REASON3D_CKPT` — checkpoint to load (`model.reason3d_checkpoint`).
- **Dataset key:** `3d_refer`. **PTH:** `REASON3D_PTH_SUBDIR` (default `processed_surprise_full_pth`, always passed) and optional `REASON3D_PTS_ROOT`.
- **Optional:** `REASON3D_SAVE_PREDS=1|0` toggles `run.save_eval_predictions`; `REASON3D_FILTER_MISSING_GT_IN_PTH=0`; `REASON3D_EVAL_RESUME=1` with `REASON3D_EVAL_JOB_ID` to resume qualitative JSONL.
- **Multi-GPU:** `NPROC` &gt; 1 → `torchrun` + `run.distributed=true` + `run.use_dist_eval_sampler=true`.

### `run_surprise_zeroshot_eval_small.sh`

- Same env pattern as full eval, **`3d_refer`**, default **`reason3d_surprise_zeroshot_small.yaml`** (e.g. `eval_scene_allowlist`). No `eval_max_samples` in YAML; cap via `--options` if needed.
- **Multi-GPU:** same as full eval script.

### `run_surprise_zeroshot_eval_small_geo.sh`

- **Required:** `REASON3D_CKPT` (finetune geo checkpoint).
- **Dataset key:** `3d_refer_geo`. **Same eval data scope** as `run_surprise_zeroshot_eval_small.sh`: `surprise_val.json` is restricted to **only** the scene IDs in `scripts/trial_scenes.txt` (3 scenes). Neither small YAML sets `eval_max_samples`; both run **all** QAs for those scenes. **PTH / filter:** `REASON3D_PTH_SUBDIR` (default `processed_surprise_full_pth`), optional `REASON3D_PTS_ROOT`, and `REASON3D_FILTER_MISSING_GT_IN_PTH` (default on). **Extra** **`REASON3D_EVAL_OPTIONS`** for more overrides. `reason3d_surprise_finetune_geo_quick.yaml` uses the same `geo_relational_cfg` as the full `reason3d_surprise_finetune_geo.yaml`, so the small-geo val YAML matches both.
- **Resume:** `REASON3D_EVAL_RESUME` + `REASON3D_EVAL_JOB_ID`; `REASON3D_SAVE_PREDS=0` can force-disable saving.
- **Note:** This wrapper **only runs single-process** `python evaluate.py` (no `NPROC` / `torchrun` in the script). For multi-GPU geo eval you would need to extend it similarly to the non-geo eval scripts or call `evaluate.py` manually with `run.distributed=true` and `run.use_dist_eval_sampler=true`.

---

## Related helper (not train/eval)

- **`run_visualize_qualitative.sh`:** Requires `QUAL_DIR` (or first arg) pointing at a `.../qualitative` folder from an eval run; uses `REASON3D_PTS_ROOT`, `REASON3D_PTH_SUBDIR`. See `visualize_qualitative_preds.py`.
- **CUDA build:** `build_pointgroup_ops.sh` — required before imports / `evaluate.py` (and training).

---

## YAML configs in-repo without dedicated bash wrappers

| File | Purpose |
|------|--------|
| `lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml` | ScanRefer training scratch |
| `lavis/projects/reason3d/val/reason3d_scanrefer_scratch.yaml` | ScanRefer val / eval |
| `lavis/projects/reason3d/train/reason3d_surprise_finetune_geo_quick.yaml` | Short geo finetune (subset/epochs; same `geo_relational_cfg` and `pth_rel_subdir` as full geo) |

Use: `python train.py --cfg-path ...` or `python evaluate.py --cfg-path ... --options ...` with the same `PYTHONPATH` pattern as the shell scripts.

---

## Pairing baselines: train script → eval script (conceptual)

| Train | Natural eval (same stack) |
|--------|---------------------------|
| `run_surprise_finetune.sh` | `run_surprise_zeroshot_eval.sh` and/or `run_surprise_zeroshot_eval_small.sh` |
| `run_surprise_finetune_geo.sh` (default `reason3d_surprise_finetune_geo.yaml`) | `run_surprise_zeroshot_eval_small_geo.sh` with val YAML `geo_relational_cfg` matching training |

**Rule of thumb:** `model.geo_relational_cfg` in the eval YAML must match the finetune YAML that produced `REASON3D_CKPT` (or pass matching `--options`).

---

*Last reviewed against `scripts/*.sh` in this repository layout; re-check scripts if you add new runners or change defaults.*
