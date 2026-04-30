# Chain v3 ablation tracker (loss-only round)

Companion to:
- design — `chainv3_design_proposal.md` (loss derivations §2, ablation plan §6)
- literature — `chainv3_literature_sweep.md`
- architecture narrative — `Models/reason3d/scripts/architecture_reason3d_chainv3.py`

All runs use the same backbone, Q-Former, Flan-T5, mask decoder, and chain-style
text targets. Only the segmentation criterion (`CriterionV3`) and dataset
emission (`ThreeDReferDatasetChainV3` adds per-instance GT) differ. Eval task
`3d_refer_seg_v3` reports the union mIoU / Acc as before, **plus**
per-instance `meanMaxIoU` / `hit@0.25` / `hit@0.50`.

> Convention: `Acc@τ` is the union (single-mask) accuracy already produced by
> `ThreeDReferSegTask`. `hit@τ` is the new per-instance metric (any GT
> instance counts as a hit). For single-target queries the two collapse to
> the same number; the gap is the multi-target story.

---

## Table A — Plan (columns describe the configuration of each run)

| #   | Run name             | Description                                                              | best_of_set | scale_aware | boundary (Lovasz) | point_aux | small_size_threshold | lambda_cov | Init ckpt baseline                        | Notes |
|-----|----------------------|--------------------------------------------------------------------------|:-----------:|:-----------:|:-----------------:|:---------:|:--------------------:|:----------:|-------------------------------------------|-------|
| A0  | `v0_chain_v2`        | Existing chain v2 baseline (no v3 loss). Provides the reference numbers. |      —      |      —      |         —         |     —     |          —           |     —      | `reason3d_inference.pth`                  | Already trained; copy results in. |
| A1  | `v3_default`         | best-of-set + scale-aware (loss-only headline run).                      |   ✅       |   ✅       |        ❌         |    ❌    |          50          |   0.10     | same                                      | Default in `train/reason3d_surprise_finetune_chainv3.yaml`. |
| A2  | `v3_no_bos`          | Disable best-of-set; keep scale-aware. Isolates small-object benefit.    |   ❌       |   ✅       |        ❌         |    ❌    |          50          |   0.10     | same                                      | If A2 ≈ A0 we know best-of-set is the multi-target lever. |
| A3  | `v3_no_scale`        | Best-of-set on, scale-aware off. Isolates multi-target benefit.          |   ✅       |   ❌       |        ❌         |    ❌    |          50          |   0.10     | same                                      | Pair with A2 for the 2×2. |
| A4  | `v3_lovasz`          | Default + Lovasz boundary on small instances.                            |   ✅       |   ✅       |        ✅         |    ❌    |          50          |   0.10     | same                                      | Watch small-object hit@0.50. |
| A5  | `v3_pointaux`        | Default + focal point-aux on small instances.                            |   ✅       |   ✅       |        ❌         |   ✅     |          50          |   0.10     | same                                      | More point-level supervision; little extra cost. |
| A6  | `v3_all_loss`        | Best-of-set + scale-aware + Lovasz + point-aux.                          |   ✅       |   ✅       |        ✅         |   ✅     |          50          |   0.10     | same                                      | Headline "all-loss" row in design §6. |
| A7  | `v3_all_loss_th80`   | Sweep `small_size_threshold` 50 → 80 with all-loss.                      |   ✅       |   ✅       |        ✅         |   ✅     |          80          |   0.10     | same                                      | Cheap sensitivity check. |
| A8  | `v3_cov_off`         | Default with `lambda_cov=0`; tests if coverage hinge matters.            |   ✅       |   ✅       |        ❌         |    ❌    |          50          |   0.00     | same                                      | If A8 ≈ A1, drop the hinge. |

> Add rows below this line for any extra sweeps (longer training, larger LR,
> per-`question_type` deep dives) — keep the same columns for consistency.

| #   | Run name             | Description                                                              | best_of_set | scale_aware | boundary (Lovasz) | point_aux | small_size_threshold | lambda_cov | Init ckpt baseline                        | Notes |
|-----|----------------------|--------------------------------------------------------------------------|:-----------:|:-----------:|:-----------------:|:---------:|:--------------------:|:----------:|-------------------------------------------|-------|
|     |                      |                                                                          |             |             |                   |           |                      |            |                                           |       |

---

## Table B — Tracker (columns describe the run as it executes / completes)

Fill in `Job ID`, `Started`, `Wall clock`, `Output dir`, the metric columns,
and `Status` (`queued / running / done / failed / aborted`).

| #   | Run name             | Job ID | Output dir                                | Started (UTC)        | Wall clock | Best ckpt epoch | mIoU | Acc@0.25 | Acc@0.50 | meanMaxIoU | hit@0.25 | hit@0.50 | Status | Notes / link to log |
|-----|----------------------|--------|-------------------------------------------|----------------------|-----------:|:---------------:|-----:|---------:|---------:|-----------:|---------:|---------:|--------|---------------------|
| A0  | `v0_chain_v2`        |        | `output/reason3d_surprise_finetune_chain_v2` |                      |            |                 |      |          |          |            |          |          |        | Pre-existing run.   |
| A1  | `v3_default`         |        | `output/reason3d_surprise_finetune_chainv3` |                      |            |                 |      |          |          |            |          |          |        |                     |
| A2  | `v3_no_bos`          |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |
| A3  | `v3_no_scale`        |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |
| A4  | `v3_lovasz`          |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |
| A5  | `v3_pointaux`        |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |
| A6  | `v3_all_loss`        |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |
| A7  | `v3_all_loss_th80`   |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |
| A8  | `v3_cov_off`         |        |                                           |                      |            |                 |      |          |          |            |          |          |        |                     |

---

## Optional — per-`question_type` breakdown (fill once main table is done)

`question_type` values from Surprise3D: `relative_position`, `narrative_perspective`,
`parametric_perspective`, `absolute_distance`. Use `eval_save_predictions=true`
in the val YAML and re-run `analyze_surprise100_attribution.py` (or
`summarize_surprise_predictions.py`) to compute the per-type splits.

| #   | Run name             | rel_pos mIoU / hit@0.5 | narr_persp mIoU / hit@0.5 | param_persp mIoU / hit@0.5 | abs_dist mIoU / hit@0.5 | Notes |
|-----|----------------------|------------------------|---------------------------|----------------------------|-------------------------|-------|
| A0  | `v0_chain_v2`        |                        |                           |                            |                         |       |
| A1  | `v3_default`         |                        |                           |                            |                         |       |
| ... | ...                  |                        |                           |                            |                         |       |

---

## Command templates (copy-paste, set `NPROC` / `Job ID` per run)

All commands assume `cd Models/reason3d` and a working
`scripts/conda_init_reason3d.sh` (already sourced from the wrapper).

> **Save-vs-eval order** (epoch-based `RunnerBase`, post-chain-v3 patch):
> the order is now `train_epoch → _save_checkpoint(epoch) → eval_epoch (per
> split in valid_splits)`. Save runs *before* eval, so an OOM during eval
> cannot lose the just-trained weights. Toggle the legacy
> `train → eval → save` order with `run.save_before_eval=false`.
>
> **Mid-epoch saves**: `run.save_every_n_steps=N` (0 = off, default) writes
> a `checkpoint_ep<E>_iter<I>.pth` every N optimizer steps. Resume from
> exactly that step with
> `REASON3D_RESUME_CKPT=output/.../checkpoint_ep<E>_iter<I>.pth` — the
> runner reads `epoch` and `iters` from the ckpt, sets
> `start_epoch = E`, and skips the first `I+1` batches of that epoch.
> DataLoader shuffle order is not checkpointed, so post-resume sample
> order differs slightly from the original run; training proceeds forward
> from the saved optimizer/model state.
>
> **Checkpoint filenames** under `RunnerBase`:
>
> | Pattern                                | When                                   |
> |----------------------------------------|----------------------------------------|
> | `checkpoint_<E>.pth`                   | end of every epoch (`E=0..max_epoch-1`) |
> | `checkpoint_ep<E>_iter<I>.pth`         | every `save_every_n_steps` mid-epoch    |
> | `checkpoint_best.pth`                  | *never written by RunnerBase* (only `RunnerIter`) |
>
> Use the last-epoch `checkpoint_<max_epoch-1>.pth` (or any specific epoch
> checkpoint) when filling `REASON3D_CKPT` for the eval step.

The `output_dir` defaults to `output/reason3d_surprise_finetune_chainv3`; for
runs A2..A8 override it via `REASON3D_TRAIN_OPTIONS` so logs / checkpoints
don't clobber A1.

```bash
# A1 — v3_default (loss-only headline)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_default" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A2 — v3_no_bos
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_no_bos model.seg_criterion_cfg.enable_best_of_set=false" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A3 — v3_no_scale
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_no_scale model.seg_criterion_cfg.enable_scale_aware=false" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A4 — v3_lovasz (default + Lovasz boundary)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_lovasz model.seg_criterion_cfg.enable_boundary=true" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A5 — v3_pointaux (default + focal point-aux)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_pointaux model.seg_criterion_cfg.enable_point_aux=true" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A6 — v3_all_loss (best-of-set + scale-aware + Lovasz + point-aux)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_all_loss model.seg_criterion_cfg.enable_boundary=true model.seg_criterion_cfg.enable_point_aux=true" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A7 — v3_all_loss_th80 (sweep small_size_threshold 50 → 80)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_all_loss_th80 model.seg_criterion_cfg.enable_boundary=true model.seg_criterion_cfg.enable_point_aux=true model.seg_criterion_cfg.small_size_threshold=80" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

```bash
# A8 — v3_cov_off (lambda_cov = 0)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_cov_off model.seg_criterion_cfg.lambda_cov=0.0" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

### Mid-epoch checkpointing + resume (recommended for long runs)

Append `run.save_every_n_steps=<N>` to `REASON3D_TRAIN_OPTIONS` to write
`checkpoint_ep<E>_iter<I>.pth` every N optimizer steps. With
`batch_size_train=2` and ~20k Surprise train rows, `N=2000` ≈ 1 save every
4000 samples (≈ 1/2.5 of a 10k-iter epoch).

```bash
# A1 — v3_default with mid-epoch saves every 2000 steps
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_default run.save_every_n_steps=2000" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

After a crash (OOM, preemption, NCCL hang, etc.), find the latest mid-epoch
file under the run's `output_dir` and pass it as `REASON3D_RESUME_CKPT`:

```bash
# Same command as the original launch, plus REASON3D_RESUME_CKPT pointing at
# the most recent mid-epoch (or per-epoch) ckpt.
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_RESUME_CKPT=output/reason3d_surprise_finetune_chainv3_default/checkpoint_ep0003_iter00006000.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_default run.save_every_n_steps=2000" \
  bash scripts/run_surprise_finetune_chainv3.sh
```

> Tip: keep `output_dir` identical between the original and the resume
> launch; otherwise the runner writes to a fresh dir and the new
> `checkpoint_<E>.pth` files coexist with the old ones in different
> folders. The script wrapper builds the `output_dir` path itself, so set
> it via `run.output_dir=…` in `REASON3D_TRAIN_OPTIONS`.
>
> For a per-epoch (no mid-epoch) resume just point
> `REASON3D_RESUME_CKPT=output/.../checkpoint_3.pth` (any epoch); the
> runner sets `start_epoch = 4` (epoch+1) and starts a fresh epoch.

### Eval-only (after each finetune; runs the v3 task with hit@τ)

`max_epoch=20` in the training YAML produces `checkpoint_0.pth` ... `checkpoint_19.pth`.
Pick the last (or any specific) one:

```bash
CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
  REASON3D_CKPT=output/reason3d_surprise_finetune_chainv3_default/checkpoint_19.pth \
  bash scripts/run_surprise_zeroshot_eval.sh
```

> Re-run the eval once with `save_eval_predictions=true` if you want the
> `predictions.jsonl` file for the per-`question_type` Table-C breakdown.

---

## How to recover the metric numbers from a finished run

The v3 eval task (`3d_refer_seg_v3`) prints a clearly delimited block at the
end of each evaluation; look for `chainv3 metrics` in the stdout / `train.log`
of the run:

```
==============================================================================
                          chainv3 metrics
==============================================================================
split=test  epoch=4  n_union=12345  n_per_inst=12345  output_dir=output/...

Union (single mask):        mIoU=0.4531  Acc@0.25=0.5012  Acc@0.50=0.3210
Per-instance (best of GT):  meanMaxIoU=0.5234  hit@0.25=0.6012  hit@0.50=0.4123

Val v3 per-instance: meanMaxIoU/hit@0.25/hit@0.50 0.5234/0.6012/0.4123 (n=12345)

Tracker row (chainv3_ablation_tracker.md Table B, metric cells only):
  | epoch | mIoU | Acc@0.25 | Acc@0.50 | meanMaxIoU | hit@0.25 | hit@0.50 |
  | 4 | 0.4531 | 0.5012 | 0.3210 | 0.5234 | 0.6012 | 0.4123 |

Full tracker row (fill in #/run-name/job-id/started/wall-clock/status/notes):
  | _#_ | _run-name_ | _jobid_ | output/... | _started_ | _wall_ | 4 | 0.4531 | 0.5012 | 0.3210 | 0.5234 | 0.6012 | 0.4123 | done | _notes_ |
==============================================================================
Wrote output/.../metrics_v3_test.json
```

Copy the metric-cells row (or the full row) into Table B above. The
`metrics_v3_<split>.json` file written next to the run output has the same
numbers in machine-readable form (see `tracker_row_metric_cells`).

For per-`question_type` (Table C), re-run eval with
`save_eval_predictions=true` so `qualitative/predictions.jsonl` is populated,
then run `scripts/summarize_surprise_predictions.py` over it (extend that
script to also aggregate `hit_at_25` / `hit_at_50` / `max_per_instance_iou`
from the per-row fields).

---

## Eval script (which one and how to run it)

The eval entry point is the existing
[`scripts/run_surprise_zeroshot_eval.sh`](../scripts/run_surprise_zeroshot_eval.sh);
it has been extended so that when `CFG=*zeroshot_chainv3.yaml` is selected,
the env-var overrides (`REASON3D_PTH_SUBDIR`, `REASON3D_PTS_ROOT`,
`REASON3D_FILTER_MISSING_GT_IN_PTH`) are routed to the
`datasets.3d_refer_chainv3.*` keys instead of the bare `3d_refer.*` keys.
The chainv3 val YAML is [`lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml`](../lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml);
it pins `task: 3d_refer_seg_v3`, `arch: reason3d_t5_chainv3`, builder
`3d_refer_chainv3`, and the v3 `seg_criterion_cfg` knobs (which only matter
for completeness — eval does not call the loss).

> The existing chain v2 eval YAML (`reason3d_surprise_zeroshot_chain.yaml`)
> still works and reports the union mIoU / Acc only. Switching to the chainv3
> YAML is what gives you the per-instance `hit@τ` / `meanMaxIoU` columns.

### Eval-only command template (all ablation rows use the same form)

`RunnerBase` doesn't track best, so use the last-epoch checkpoint
(`checkpoint_<max_epoch-1>.pth`) — or any specific epoch you want to eval:

```bash
# Single-GPU eval (default)
CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
  REASON3D_CKPT=output/reason3d_surprise_finetune_chainv3_default/checkpoint_19.pth \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  bash scripts/run_surprise_zeroshot_eval.sh
```

```bash
# Multi-GPU eval (shards the test set across NPROC ranks; identical metrics).
NPROC=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
  REASON3D_CKPT=output/reason3d_surprise_finetune_chainv3_default/checkpoint_19.pth \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  bash scripts/run_surprise_zeroshot_eval.sh
```

```bash
# With per-row JSONL (needed for Table C per-question_type breakdown).
CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
  REASON3D_CKPT=output/reason3d_surprise_finetune_chainv3_default/checkpoint_19.pth \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_SAVE_PREDS=1 \
  bash scripts/run_surprise_zeroshot_eval.sh
```

### Output locations after eval

Relative to the val YAML's `run.output_dir`
(default: `output/reason3d_surprise_zeroshot_chainv3`):

| Path                                       | When                          | Contents                                            |
|--------------------------------------------|-------------------------------|-----------------------------------------------------|
| `train.log` (stdout)                       | always                        | The `chainv3 metrics` block above; copy from there. |
| `metrics_v3_test.json`                     | always (main rank)            | Machine-readable mIoU/Acc/meanMaxIoU/hit@τ + tracker_row string. |
| `qualitative/predictions.jsonl`            | only if `REASON3D_SAVE_PREDS=1` | One row per sample with `point_iou`, `max_per_instance_iou`, `hit_at_25`, `hit_at_50`, `question_type`, `decoded_text`, `text_input`, `scene_id`, `ann_id`. |
| `qualitative/masks/<scene>_<ann>_*.npz`    | only if `REASON3D_SAVE_EVAL_MASKS=1` | Per-row `pred_pmask` and `gt_pmask` arrays (large on full val).  |

### Per-row table for evaluating one ablation row

For each finished training run in Table B:

1. Locate the latest checkpoint under its `Output dir`
   (`output/reason3d_surprise_finetune_chainv3_*/checkpoint_<max_epoch-1>.pth`,
   e.g. `checkpoint_19.pth` for `max_epoch=20`). `RunnerBase` produces one
   `checkpoint_<N>.pth` per epoch and never a `checkpoint_best.pth`.
2. Launch the eval command above with `REASON3D_CKPT=<that-path>`.
3. Wait for the `chainv3 metrics` block to print.
4. Copy the metric-cells row into Table B for that ablation row.
5. (Optional) Diff `metrics_v3_test.json` across runs for a one-screen
   summary.

### Re-running eval after training already produced metrics

The training YAML keeps `valid_splits: []` so the in-line eval is skipped at
training time (faster wall clock, like chain v2). The numbers for Table B
come exclusively from the post-hoc eval invocation above. If you want the
training loop to also run periodic eval, override
`run.valid_splits=["val"]` in `REASON3D_TRAIN_OPTIONS` — the eval at each
checkpoint will then print the same `chainv3 metrics` block.
