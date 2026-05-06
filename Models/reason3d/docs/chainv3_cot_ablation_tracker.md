# Chain v3 CoT ablation tracker (architectural round)

Companion to:
- design — `chainv3_cot_design_space.md` (axes 1–3 + tactical A–H, B-row grid §"Proposed ablation rows")
- literature — `chainv3_cot_literature_review.md`
- loss-only round — `chainv3_ablation_tracker.md` (A0–A8)
- model — `lavis/models/reason3d_models/reason3d_t5_chainv3_cot.py` (`Reason3DT5ChainV3CoT`)
- dataset — `lavis/datasets/datasets/threedrefer_datasets_chainv3_cot.py` (`ThreeDReferDatasetChainV3CoT`)
- train YAML — `lavis/projects/reason3d/train/reason3d_surprise_finetune_chainv3_cot.yaml`

All B-rows inherit the **CriterionV3 default** (best-of-set BCE+Dice + scale-aware
dice; boundary / point-aux off; `small_size_threshold=50`, `lambda_cov=0.10`)
unless explicitly noted. The loss-side and CoT grids are orthogonal but
composable; the combined-system row in the report is `A6 ⊕ B1'`.

> Convention: same as the loss-only tracker. `Acc@τ` = union (single-mask)
> accuracy; `hit@τ` = per-instance (any-GT-instance hit). Both are produced by
> the v3 eval task `3d_refer_seg_v3`.

---

## Table A — Plan (columns describe the configuration of each run)

| #     | Run name                       | Description                                                                                  | cot_template_prob | template family | mass-pool feedback (F1) | stop-grad on M₁ (G=1b) | aux M₁ rewards (W1) | anti-collapse | question-side prefix | Init ckpt baseline                       | Notes |
|-------|--------------------------------|----------------------------------------------------------------------------------------------|:-----------------:|:---------------:|:-----------------------:|:----------------------:|:-------------------:|:-------------:|:--------------------:|------------------------------------------|-------|
| B0    | `cot_off`                      | Single-`[SEG]` chain-v2 (loss-only). Sanity floor; should reproduce A1 numbers.              |        0.0        |       —         |           —             |           —            |          —          |       —       |          —           | `reason3d_inference.pth`                 | Two-pass code path inactive (no `cot_rows`). Equivalent to A1 with the cot model class. |
| B1'   | `cot_pure` *(headline)*        | M2 + W1-pure + F1 + P4 + two-pass (G=1b, H=2a). All locked claims together.                  |        1.0        |    P4 (5×)      |          on             |          on            |        none         |      off      |         off          | same                                     | The headline of this round. |
| B2    | `cot_p2_stilted`               | P2 (`"First, the {landmark}. [SEG]. Then... [SEG]."`) instead of P4.                         |        1.0        |    P2 (1×)      |          on             |          on            |        none         |      off      |         off          | same                                     | Tests whether the natural P4 phrasing matters vs the stilted regex-style template. |
| B3    | `cot_with_prefix`              | Question-side reasoning prefix on (`"Let's think step by step. "` injected ahead of `Q`).    |        1.0        |    P4 (5×)      |          on             |          on            |        none         |      off      |         **on**       | same                                     | Tests Kojima-2022 zero-shot trigger; train+eval match. |
| B5    | `cot_p3_steps`                 | P3 (`"Step 1: ... [SEG]. Step 2: ... [SEG]."`) instead of P4.                                |        1.0        |    P3 (1×)      |          on             |          on            |        none         |      off      |         off          | same                                     | Tests structured-step phrasing vs natural language. |
| B6    | `cot_with_rewards`             | Add containment + cosine consistency rewards on M₁ (the original W1 stack).                  |        1.0        |    P4 (5×)      |          on             |          on            |        **on**       |      off      |         off          | same                                     | Tests whether explicit consistency rewards add anything beyond architectural F1. |
| B6+   | `cot_grad_on_M1`               | Axis G flipped to 1a: gradient flows through `mask_pool_token` back into M₁.                 |        1.0        |    P4 (5×)      |          on             |         **off**        |        none         |      off      |         off          | same                                     | Tests indirect "grade M₁ via M₂'s loss" channel. |
| B7    | `cot_no_arch_feedback`         | F3 (no `MaskPoolToken`, no two-pass; just two `[SEG]`s in one forward, text-CoT only).       |        1.0        |    P4 (5×)      |        **off**          |          —             |        none         |      off      |         off          | same                                     | Isolates the architectural feedback contribution from text-CoT alone. |
| B7+   | `cot_full_old_headline`        | F1 + W1 rewards + anti-collapse + grad-on (the *previously proposed* B1).                    |        1.0        |    P4 (5×)      |          on             |         **off**        |        **on**       |    **on**     |         off          | same                                     | Documents what the old W1 headline scored; auxiliary-loss upper bound. |
| B8    | `cot_p4_fixed_prob`            | `cot_template_prob = 0.5` instead of regex-hit-gated 1.0; landmark-fallback uses target NP.  |        0.5        |    P4 (5×)      |          on             |          on            |        none         |      off      |         off          | same                                     | Tests whether the regex gate matters vs adds complexity. |
| B9    | `cot_llm_rationale` (Tier 2)   | Replace per-sample answer with offline LLM-generated task-adaptive rationale (variable `[SEG]` count). |     1.0     |    LLM-mined    |          on             |          on            |        none         |      off      |         off          | B1' ckpt (preferred) or `reason3d_inference.pth` | Phase 2; covers ~80% of non-landmark queries (cs / hi / camera_view). |
| B11   | `cot_star` (Tier 3, deferred)  | Self-generated CoT via inference sampling + final-mask-IoU filter + fine-tune (K=2-3 rounds). |        1.0        |    self-gen     |          on             |          on            |        none         |      off      |         off          | B1' or B9 ckpt                            | Deferred. |

> Add rows below for any extra sweeps (longer training, larger LR,
> per-`question_type` deep dives, ckpt-from-A1 vs ckpt-from-baseline) — keep
> the same columns for consistency.

| #     | Run name                       | Description                                                                                  | cot_template_prob | template family | mass-pool feedback (F1) | stop-grad on M₁ (G=1b) | aux M₁ rewards (W1) | anti-collapse | question-side prefix | Init ckpt baseline                       | Notes |
|-------|--------------------------------|----------------------------------------------------------------------------------------------|:-----------------:|:---------------:|:-----------------------:|:----------------------:|:-------------------:|:-------------:|:--------------------:|------------------------------------------|-------|
|       |                                |                                                                                              |                   |                 |                         |                        |                     |               |                      |                                          |       |

---

## Table B — Tracker (columns describe the run as it executes / completes)

Fill in `Job ID`, `Started`, `Wall clock`, `Output dir`, the metric columns,
and `Status` (`queued / running / done / failed / aborted`). The metric block
matches the loss-only tracker so a row from `metrics_v3_<split>.json` drops
straight in.

| #     | Run name                  | Job ID | Output dir                                          | Started (UTC) | Wall clock | Best ckpt epoch | mIoU | Acc@0.25 | Acc@0.50 | meanMaxIoU | hit@0.25 | hit@0.50 | Status | Notes / link to log |
|-------|---------------------------|--------|-----------------------------------------------------|---------------|-----------:|:---------------:|-----:|---------:|---------:|-----------:|---------:|---------:|--------|---------------------|
| B0    | `cot_off`                 |        | `output/reason3d_surprise_finetune_chainv3_cot_off`  |               |            |                 |      |          |          |            |          |          |        | Should reproduce A1 within run-to-run variance. |
| B1'   | `cot_pure`                |        | `output/reason3d_surprise_finetune_chainv3_cot`      |               |            |                 |      |          |          |            |          |          |        | Headline. |
| B2    | `cot_p2_stilted`          |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs template-family knob. |
| B3    | `cot_with_prefix`         |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs prompt-prefix knob. |
| B5    | `cot_p3_steps`            |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs template-family knob. |
| B6    | `cot_with_rewards`        |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs W1 aux-loss module. |
| B6+   | `cot_grad_on_M1`          |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs stop-grad knob. |
| B7    | `cot_no_arch_feedback`    |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs F1-bypass knob. |
| B7+   | `cot_full_old_headline`   |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs W1 + grad-on + anti-collapse. |
| B8    | `cot_p4_fixed_prob`       |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Needs landmark-fallback knob. |
| B9    | `cot_llm_rationale`       |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Phase 2; needs offline LLM mining pipeline. |
| B11   | `cot_star`                |        |                                                     |               |            |                 |      |          |          |            |          |          |        | Deferred. |

---

## Optional — per-`question_type` breakdown (fill once main table is done)

`question_type` values from Surprise3D: `relative_position`, `abs`, `first_view`,
`camera_view`, `cs`, `hi`. The CoT mechanism is expected to land on the
**relational subset** (`relative_position`, `abs`, `first_view`); regressions
on `cs` / `hi` would mean the multi-`[SEG]` training is hurting the
non-landmark majority — that's exactly the question B1' vs B0 is set up to
answer.

| #     | Run name                  | rel_pos hit@0.5 | abs hit@0.5 | first_view hit@0.5 | camera_view hit@0.5 | cs hit@0.5 | hi hit@0.5 | Notes |
|-------|---------------------------|-----------------|-------------|--------------------|---------------------|------------|------------|-------|
| B0    | `cot_off`                 |                 |             |                    |                     |            |            | Floor — no CoT effect expected. |
| B1'   | `cot_pure`                |                 |             |                    |                     |            |            | Win on rel_pos / abs / first_view; flat-or-down on cs / hi is acceptable. |
| ...   | ...                       |                 |             |                    |                     |            |            |       |

---

## Code-ready vs. requires-toggle status

| #     | Run name                  | Code-ready? | What's missing                                                                                          |
|-------|---------------------------|:-----------:|---------------------------------------------------------------------------------------------------------|
| B0    | `cot_off`                 |   ✅        | Set `datasets.3d_refer_chainv3_cot.cot_template_prob=0.0` in `REASON3D_TRAIN_OPTIONS`.                  |
| B1'   | `cot_pure` (headline)     |   ✅        | Default config; no overrides needed.                                                                    |
| B2    | `cot_p2_stilted`          |   ⚠️         | Needs `cot_template_family: p2` in dataset (add a P2 list + selector to `_build_cot_answer`).            |
| B3    | `cot_with_prefix`         |   ⚠️         | Needs `cot_prompt_prefix: bool` in dataset (prepend `"Let's think step by step. "` to the question).    |
| B5    | `cot_p3_steps`            |   ⚠️         | Needs `cot_template_family: p3` in dataset (same selector as B2).                                       |
| B6    | `cot_with_rewards`        |   ⚠️         | Needs W1 aux losses (`L_contain` + cosine) on M₁ in `Reason3DT5ChainV3CoT.forward`; gated by `model.cot_aux_w1_rewards`. |
| B6+   | `cot_grad_on_M1`          |   ⚠️         | Needs `model.cot_stop_grad_on_pool: bool = true`; flip to `false` to drop the `.detach()`.              |
| B7    | `cot_no_arch_feedback`    |   ⚠️         | Needs `model.cot_arch_feedback: bool = true`; flip to `false` to skip the F1 mass-pool injection.       |
| B7+   | `cot_full_old_headline`   |   ⚠️         | Needs B6 + B6+ + `model.cot_anti_collapse: bool` (PixelLM-style margin).                                |
| B8    | `cot_p4_fixed_prob`       |   ⚠️         | Needs `cot_landmark_fallback: target_np` in dataset (use the target noun phrase as a fallback when regex misses). |
| B9    | `cot_llm_rationale`       |   ❌        | Needs offline LLM mining pipeline + JSON schema for variable-`[SEG]` rationale + a `ThreeDReferDatasetChainV3CoT` subclass that loads pre-computed rationales. |
| B11   | `cot_star`                |   ❌        | Needs sampling + IoU-filter + iterative-finetune harness on top of B1' or B9. Deferred to phase 3.       |

The "⚠️" rows each cost a single small flag in the YAML and a 5–20-line patch
in the dataset / model. They are intentionally **not** wired up yet so the
B1' headline lands on a frozen, minimal code surface — every additional knob
is a new variable in the regression.

---

## Run launch templates (B0, B1' — code-ready today)

The `output_dir` defaults to `output/reason3d_surprise_finetune_chainv3_cot`;
override it via `REASON3D_TRAIN_OPTIONS` for each row so logs / checkpoints
don't clobber the headline run.

```bash
# B1' — cot_pure (headline)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_cot" \
  bash scripts/run_surprise_finetune_chainv3_cot.sh
```

```bash
# B0 — cot_off (single-[SEG] floor; should reproduce A1)
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_cot_off datasets.3d_refer_chainv3_cot.cot_template_prob=0.0" \
  bash scripts/run_surprise_finetune_chainv3_cot.sh
```

### Mid-epoch checkpointing + resume

Same recipe as the loss-only round — append `run.save_every_n_steps=<N>`
to `REASON3D_TRAIN_OPTIONS` and use `REASON3D_RESUME_CKPT` for restarts.
See `chainv3_ablation_tracker.md` §"Mid-epoch checkpointing + resume".

### Eval-only (after each finetune)

A `reason3d_surprise_zeroshot_chainv3_cot.yaml` val YAML still needs to be
created; the recipe is identical to the chainv3 val YAML with two changes:

```yaml
model:
  arch: reason3d_t5_chainv3_cot
datasets:
  3d_refer_chainv3_cot:        # not 3d_refer_chainv3
    type: 3d_refer_chainv3_cot
    ...
run:
  task: 3d_refer_seg_v3        # same v3 task; produces hit@τ
```

Once it exists, the eval command is the same template as the chainv3 round:

```bash
CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3_cot.yaml \
  REASON3D_CKPT=output/reason3d_surprise_finetune_chainv3_cot/checkpoint_19.pth \
  bash scripts/run_surprise_zeroshot_eval.sh
```

> Recovering metrics from a finished run, exporting per-`question_type`
> predictions, and the `chainv3 metrics` stdout block all behave exactly as
> in the loss-only tracker — see `chainv3_ablation_tracker.md`
> §"How to recover the metric numbers from a finished run".

---

## Required eval-script extensions

The eval entry point `scripts/run_surprise_zeroshot_eval.sh` already routes
`REASON3D_PTH_SUBDIR` / `REASON3D_PTS_ROOT` / `REASON3D_FILTER_MISSING_GT_IN_PTH`
to the `datasets.3d_refer_chainv3.*` keys when `CFG=*zeroshot_chainv3.yaml`
is selected. Extend that conditional to also match
`*zeroshot_chainv3_cot.yaml` and route to `datasets.3d_refer_chainv3_cot.*`
keys before B-row evals run.

---

## Cross-grid table (combined-system row)

The single combined row in the report is `A6 ⊕ B1'` — best loss stack
(`enable_boundary=true`, `enable_point_aux=true`) with the headline CoT
recipe (W1-pure + F1 + P4 + two-pass + stop-grad). Launch:

```bash
# A6 ⊕ B1' — combined system (loss A6 + CoT B1')
NPROC=4 \
  REASON3D_PTS_ROOT=/nfs-stor/lan.wei/data/scannetpp/ \
  REASON3D_PTH_SUBDIR=processed_surprise_full_pth \
  REASON3D_INIT_CKPT=/nfs-stor/lan.wei/data/checkpoints/reason3d_inference.pth \
  REASON3D_TRAIN_OPTIONS="run.output_dir=output/reason3d_surprise_finetune_chainv3_cot_a6_combined model.seg_criterion_cfg.enable_boundary=true model.seg_criterion_cfg.enable_point_aux=true" \
  bash scripts/run_surprise_finetune_chainv3_cot.sh
```

| #          | Run name                | Loss config (CriterionV3 knobs)                                          | CoT config            | Notes |
|------------|-------------------------|--------------------------------------------------------------------------|-----------------------|-------|
| A6 ⊕ B1'   | `cot_a6_combined`       | best-of-set + scale-aware + Lovasz boundary + focal point-aux            | B1' (P4, W1-pure, F1) | Reported as the combined-system row in the paper. |

---

## Status (this round)

- B0 / B1' — code-ready; can launch immediately on the cluster.
- B2 / B3 / B5 / B6 / B6+ / B7 / B7+ / B8 — each needs one small toggle
  (column "What's missing"); add them only after B1' has a stable number,
  to keep the headline ckpt on a frozen code surface.
- B9 — phase 2; reserved for after the regex-only B1' lands.
- B11 — deferred.
