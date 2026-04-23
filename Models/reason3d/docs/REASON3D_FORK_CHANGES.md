# Reason3D fork changes (for migration / PR description)

Summary of edits under `Models/reason3d/` for **Surprise eval**, **finetuning**, **dataset filtering**, **checkpoints**, and **qualitative dumps**.

For a **narrative** of bugs and fixes (especially **annotation `object_id` vs `sampled_instance_anno_id` / top-instance policy**), see **`REASON3D_PROBLEMS_AND_FIXES.md`** in this folder.

## Modified files (tracked)

| File | Purpose |
|------|--------|
| `lavis/configs/datasets/3dseg/defaults.yaml` | Empty `dataset_init: {}` placeholder for YAML merge. |
| `lavis/datasets/builders/base_dataset_builder.py` | Forwards `dataset_init` kwargs into dataset constructors. |
| `lavis/datasets/datasets/threedrefer_datasets.py` | `filter_missing_gt_in_pth`, `pth_rel_subdir`, eval scene allowlist / `eval_max_samples`, faster-filter logging, `get_sp_filenames` scoped to allowlist, `_pth_inst_cache`, overlap checks, `super().__init__` fix. |
| `lavis/models/base_model.py` | Fix `assert` when `pretrained` is missing (`load_checkpoint_from_config`). |
| `lavis/models/reason3d_models/reason3d_t5.py` | `reason3d_checkpoint` + `load_checkpoint_from_config`; path normalization in `load_from_pretrained`; Hub BERT/T5 (no hardcoded `/workspace/...`); remove stray `ipdb`. |
| `lavis/models/reason3d_models/lib/setup.py` | Optional `POINTGROUP_CUDA_INCLUDE_DIRS` for extension includes. |
| `lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml` | Absolute SPFormer path example. |
| `lavis/projects/reason3d/val/reason3d_scanrefer_scratch.yaml` | Same. |
| `lavis/tasks/refer_seg_task.py` | `save_eval_predictions` + `qualitative/` JSONL + mask `.npz`; `before_evaluation` prep. |
| `update_superpoints.py` | (Your session) CLI / path robustness for superpoint merge — keep in sync with your local diff. |

## New files to add (untracked until `git add`)

| Path | Purpose |
|------|--------|
| `lavis/projects/reason3d/val/reason3d_surprise_zeroshot.yaml` | Surprise zero-shot eval. |
| `lavis/projects/reason3d/val/reason3d_surprise_zeroshot_small.yaml` | Small scene allowlist eval. |
| `lavis/projects/reason3d/train/reason3d_surprise_finetune.yaml` | Surprise finetune defaults + `filter_missing_gt_in_pth`. |
| `scripts/build_pointgroup_ops.sh` | CUDA/conda-aware `pointgroup_ops` build. |
| `scripts/reinstall_pointgroup_ops.sh` | Rebuild helper. |
| `scripts/run_surprise_zeroshot_eval.sh` | Eval launcher + optional `REASON3D_SAVE_PREDS`. |
| `scripts/run_surprise_zeroshot_eval_small.sh` | Small eval launcher. |
| `scripts/run_surprise_finetune.sh` | Finetune launcher (`torchrun` if `NPROC>1`). |
| `scripts/run_trial_preprocess.sh` | Trial preprocessing pipeline. |
| `scripts/trial_scenes.txt` | Scene ids aligned with Surprise val. |
| `scripts/trial_prepare_training_data.yml` | Trial prepare_training_data config. |
| `scripts/verify_*.py`, `ensure_map_benchmark_with_counts.py` | Verification / ScanNet++ prep helpers. |
| `docs/ENVIRONMENT.md`, `docs/DATA_SYNC.md`, `docs/REASON3D_PROBLEMS_AND_FIXES.md`, this file | Cluster migration + debugging narrative docs. |
| `.gitignore` | Ignore `lavis/output/`, `__pycache__`, built `.so`. |

## Git: commit and push

**Do not commit** `lavis/output/`, `__pycache__/`, or built `*.so` (ignored by `.gitignore`).

```bash
cd /path/to/spatial_reasoning
git status
git add Models/reason3d/.gitignore
git add Models/reason3d/docs/*.md
git add Models/reason3d/lavis/configs/datasets/3dseg/defaults.yaml
git add Models/reason3d/lavis/datasets/builders/base_dataset_builder.py
git add Models/reason3d/lavis/datasets/datasets/threedrefer_datasets.py
git add Models/reason3d/lavis/models/base_model.py
git add Models/reason3d/lavis/models/reason3d_models/lib/setup.py
git add Models/reason3d/lavis/models/reason3d_models/reason3d_t5.py
git add Models/reason3d/lavis/projects/reason3d/
git add Models/reason3d/lavis/tasks/refer_seg_task.py
git add Models/reason3d/scripts/
git add Models/reason3d/update_superpoints.py
git commit -m "Reason3D: Surprise eval/finetune, dataset filters, checkpoint paths, qualitative saves, cluster docs"
```

**Remotes:** this clone has `origin` → `https://github.com/liziwennba/SURPRISE3D.git` (treat as **upstream** if you fork).

### Push your work as a **GitHub fork** (recommended flow)

1. On GitHub, open **https://github.com/liziwennba/SURPRISE3D** and click **Fork** (creates `https://github.com/<you>/SURPRISE3D` or similar under your account).  
   - Alternatively create an **empty** repo and add it as a remote (no fork link in GitHub UI, but you can still push).

2. Locally, keep the original remote for pulling updates, and add **your** fork as the remote you push to:

```bash
cd /path/to/spatial_reasoning
git remote rename origin upstream
git remote add origin git@github.com:<you>/SURPRISE3D.git
# or: https://github.com/<you>/SURPRISE3D.git
git fetch upstream
git push -u origin main
```

3. Later, sync from upstream before a PR:

```bash
git fetch upstream
git merge upstream/main   # or: git rebase upstream/main
git push origin main
```

4. Open a **Pull Request** on `liziwennba/SURPRISE3D` from your fork’s `main` (GitHub will offer this after you push).

**If you prefer not to rename `origin`**, leave it as-is and only add your fork:

```bash
git remote add fork git@github.com:<you>/SURPRISE3D.git
git push -u fork main
```

Use **SSH keys** or a **personal access token** (HTTPS); GitHub no longer accepts account passwords for `git push`.
