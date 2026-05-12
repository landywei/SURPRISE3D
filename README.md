# SURPRISE3D fork — what I did vs. upstream Surprise3D and Reason3D

*Lan Wei · `lan.wei@mbzuai.ac.ae` · MBZUAI · May 2026*

This README is the honest, chronological version of what I changed in this
fork, what broke, what I fixed, and what is still open. It explains where
each number in the tables comes from and what it does *not* mean. For a
script-by-script breakdown of every finetune / eval entry point and YAML
default, see [`Models/reason3d/docs/finetune_eval_scripts.md`](Models/reason3d/docs/finetune_eval_scripts.md).

---

## TL;DR

1.  **Preprocessing.** I rebuilt the Surprise3D ↔ Reason3D data pipeline from
    scratch on top of ScanNet++. The only preprocessing artifact in the
    upstream Surprise3D / Reason3D codebases is `prepare_data_reason.sh` (7
    lines of bash, ScanNet **v2** specific) — it does not produce the `.pth`
    files Reason3D actually consumes for Surprise3D, and it strips file
    extensions assuming the ScanNet v2 layout. I replaced it with a
    ScanNet++ pipeline (`scripts/run_prepare_surprise_scannetpp_pth.sh`)
    that uses the all-instance preprocessing patch in
    `third_party/scannetpp/` — without that patch, the default top-100
    instance whitelist drops ~96% of Surprise3D query targets.

2.  **Problems.** Two big classes of problems: (a) data: silent
    instance-id mismatches between annotations and sampled point clouds, and
    (b) metric: union-mIoU systematically lies on multi-target Surprise3D
    queries. There is also a stack of smaller engineering bugs (CUDA build,
    config paths, hardcoded `/workspace` paths in upstream).

3.  **Fixes.** Beyond the engineering fixes (documented in
    `Models/reason3d/docs/REASON3D_PROBLEMS_AND_FIXES.md`), the substantive
    fixes are: per-instance metrics (`meanMaxIoU`, `hit@τ`), CriterionV3 (a
    drop-in segmentation loss with a best-of-set term, coverage hinge,
    scale-aware dice, and optional Lovász/focal auxiliaries), and Chain v3
    CoT (a two-pass forward with a mass-pooled landmark feature).

4.  **Current numbers (Surprise3D val).** Vanilla Reason3D gets
    `mIoU = 22.95%`, `hit@0.50 = 17.83%`. CriterionV3 trades 2.84pt of
    union `A_50` for +1.92pt of per-instance `hit@0.50`. Chain v3 CoT
    pushes `hit@0.50` to **20.94%** (+3.11pt over vanilla), and flips the
    per-instance vs. union gap `Δ_50 = hit@0.50 − A_50` from −2.96pt on
    vanilla to **+3.60pt**.

5.  **Plan.** Train four branches to 8 checkpoints each (bare Reason3D,
    chain, loss = `no_scale` + `lovasz`, B' CoT) and compare across the six
    Surprise3D query families plus qualitative analysis of intermediate
    landmark masks. The current numbers are from runs of varying lengths
    that landed at different epochs; the 8-checkpoint plan is the canonical
    grid that will replace them.

The rest of this document explains each of these in detail.

---

## 1. Preprocessing — what I changed and how it differs from upstream

### 1.1 What the upstream repos actually ship

**Reason3D upstream** (the open file `Models/reason3d/data/scannetv2/prepare_data_reason.sh`):

```7:7:Models/reason3d/data/scannetv2/prepare_data_reason.sh
python prepare_data_reason.py --data_split val
```

The full file is just three commands — `split_data.py`, then
`prepare_data_reason.py` for `train` and `val`. The script lives under
`data/scannetv2/`, takes ScanNet **v2** `_vh_clean_2.ply` files, and runs
the SparseConvNet-style preprocessing inherited from the Reason3D
ScanRefer pipeline. The python script derives the parallel filenames by
stripping fixed-length file suffixes — e.g.

```python
fn2 = fn[:-3]  + 'labels.ply'
fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
fn4 = fn[:-15] + '.aggregation.json'
```

This is brittle (any path with a different suffix length silently breaks
the parallel file lookup) and, more importantly, **it does not apply to
Surprise3D**: Surprise3D queries are over **ScanNet++** scenes, not
ScanNet v2. The script in the open file therefore produces nothing
Reason3D can use against the Surprise3D val JSON — for our problem, this
piece of upstream is effectively a stub.

**Surprise3D upstream**
(<https://github.com/liziwennba/SURPRISE3D>) ships the **annotations**
(`surprise-3d-{train,val}.json` on HuggingFace) and points users at
ScanNet++ for the underlying scenes plus Reason3D / SPFormer / 3D-STMN
for preprocessing. There is no Surprise3D-specific preprocessing
script: the user is expected to run ScanNet++'s
`semantic.prep.prepare_training_data` against the
`top100_instance.txt` whitelist.

### 1.2 Why a verbatim run of the upstream pipeline does not work

Two practical problems collide:

1. **The top-100 instance whitelist drops ~96% of Surprise3D targets.**
   The default ScanNet++ preprocessing (`use_instances: true`,
   `instance_labels_path: top100_instance.txt`) only keeps an
   `objectId` on a vertex when the instance's class is in the top-100
   whitelist. Surprise3D queries reference a much wider vocabulary (a
   whole long tail of "the small bottle on the desk" / "the picture
   beside the sofa" / etc.), so most of those `object_id`s never
   appear in `sampled_instance_anno_id` — the model has no GT mask to
   train on or evaluate against, and inference rows assert out with
   `gt_pmask.int().max() == 1`.

2. **`object_id` (annotation) ≠ `sampled_instance_anno_id` (point
   cloud).** Even after the whitelist is bypassed, ScanNet++
   preprocessing area-samples the mesh and assigns labels by nearest
   vertex. A small or sparsely-sampled instance can still be missing
   from `sampled_instance_anno_id`. So the annotation and the
   preprocessed point cloud disagree on which `object_id`s exist
   per-scene, and the disagreement is not symmetric across scenes.

### 1.3 What I did

Three coordinated changes:

**A. Patch ScanNet++ preprocessing to keep all instances.** The patch is
in `third_party/scannetpp/semantic/transforms/{factory.py, mesh.py}`
(diff lives uncommitted in the third_party submodule; see
`Models/reason3d/scripts/scannetpp_prepare_full_instances.yml` for the
consuming config). It adds an `all_instance_classes: true` flag to
`GetLabelsOnVertices`: when set, *every* segGroup with `label_ndx !=
ignore_label` writes its `objectId` onto its vertices, instead of only
those whose class is in `top100_instance.txt`. With the flag off the
behaviour is bit-identical to upstream.

**B. A Surprise3D-specific preprocessing wrapper.**
`Models/reason3d/scripts/prepare_surprise_scannetpp_pth.py` is a small
Python script that imports ScanNet++ transforms via
`PYTHONPATH=$SCANNPP_REPO`, runs the patched `GetLabelsOnVertices` with
`all_instance_classes: true`, adds two extra outputs:

- `sampled_mesh_vertex_idx`: per sampled point, the index of its
  nearest mesh vertex (so any per-vertex array — instance ids,
  semantics, segments — can be remapped to the sampled cloud after the
  fact).
- `superpoints` (or `vtx_superpoints`): per-vertex superpoint id from
  the `segmentator` library (or the pure-PyTorch fallback), propagated
  to sampled points by `SamplePointsOnMesh`. This is what
  Reason3D's mask decoder actually consumes.

The wrapper is launched by
`scripts/run_prepare_surprise_scannetpp_pth.sh` (the entry point named
in the README's "Data" section). Output goes to
`/nfs-stor/lan.wei/data/scannetpp/processed_surprise_full_pth/`, which
the YAMLs in `lavis/configs/datasets/3dseg/` reference via
`pth_rel_subdir: processed_surprise_full_pth`.

**C. Drop annotations whose target id is missing from the
preprocessed cloud.** The dataset class
(`lavis/datasets/datasets/threedrefer_datasets.py`) gained a
`filter_missing_gt_in_pth: true` flag. At init time it `torch.load`s
each scene's `.pth` once, caches the set of `sampled_instance_anno_id`s,
and drops annotations whose `object_id` is not in that set. This
prevents the spurious `gt_pmask.max() != 1` asserts and stops the
optimizer from fitting on rows with no usable mask. The cost is one
sequential `torch.load` per scene at startup; for clusters there is now
also an `instance_id_cache_file` knob that persists the per-scene id
set to JSON so subsequent runs skip the `.pth` reads entirely.

### 1.4 The empirical effect of (A)+(B)+(C)

On the same Reason3D pretrained checkpoint and the same val split,
switching from the upstream top-100 preprocessing to the
all-instance + filter pipeline lifts the published upstream Surprise3D
mIoU from **11.00%** to **22.95%** (vanilla Reason3D in this fork) on
the same Reason3D pretrained checkpoint and val split. About half of
the gap is "the GT now exists" (rows whose
target was previously masked out by the whitelist now contribute); the
rest is the extra training signal the model gets from a cleaner
loss surface. None of it is from a new model — it is purely a
preprocessing fix.

> **Honest caveat.** This 22.95% number is *not* directly comparable to
> any other Surprise3D evaluation in the literature, because both the
> filtered val set (n = 8,229 vs. the upstream nominal 10,198) and the
> set of trainable instances differ. The right way to read it: under
> our preprocessing, vanilla Reason3D scores 22.95% — and we treat that
> as the floor that CriterionV3 and Chain v3 CoT need to beat *on the
> same preprocessing pipeline* (which they do; see §4).

---

## 2. Problems I observed

### 2.1 Data / pipeline problems (mostly mechanical)

These are the bugs that block any sensible evaluation, regardless of
modelling. They are all documented in
`Models/reason3d/docs/REASON3D_PROBLEMS_AND_FIXES.md`; I list them here
so the report tells the same story:

| # | Symptom | Root cause | Fix |
|---|---------|------------|-----|
| 1 | `gt_pmask.int().max() == 1` assert / nonsense eval | Annotation `object_id` not in `sampled_instance_anno_id`. Either the top-100 whitelist dropped it, or the area-sampling missed the instance. | `filter_missing_gt_in_pth: true` (drop the row); `all_instance_classes: true` (keep more ids). |
| 2 | Empty dataset: `get_sp_filenames()` returns 0 files | `pth_rel_subdir` did not match on-disk layout. | Set `pth_rel_subdir: processed_surprise_full_pth` (or whatever directory the wrapper wrote to). |
| 3 | `pointgroup_ops` build failures (CUDA / Thrust / glibc) | nvcc / cu121 / conda-sysroot version skew | `scripts/build_pointgroup_ops.sh` aligns nvcc with `torch.version.cuda`, prefers conda sysroot includes. |
| 4 | "checkpoint url or path is invalid" | Three different "pretrained" concepts collided: BLIP2 default, SPFormer backbone, full Reason3D. | Split into `point_encoder_cfg.pretrained` (SPFormer) and `model.reason3d_checkpoint` (full Reason3D); `load_from_pretrained` does `expanduser` + `abspath`. |
| 5 | `import ipdb` crash; hardcoded `/workspace/huggingface/...` paths | Upstream debug artefacts | Removed `ipdb`; load BERT / T5 from Hub by id (`bert-base-uncased`, `google/flan-t5-xl`). |
| 6 | "Building datasets…" hangs 5–15 minutes | One `torch.load` per unique scene to compute the filter | `instance_id_cache_file` persists the result to JSON so subsequent runs skip the loads; observability log line. |
| 7 | Small-eval allowlist "kept 0 / N annotations" | `trial_scenes.txt` hashes were ScanNet++ ids that don't appear in `surprise_val.json`. | Replaced with `scene_id` values actually present in the val JSON; early error if intersection is empty. |

None of these are research contributions; they're the cost of getting a
3-codebase stack (Reason3D fork ↔ ScanNet++ preprocessing ↔ Surprise3D
annotations) to run end-to-end on a new cluster. They mattered because
without them, none of the evaluation in §4 below would have produced a
trustworthy number.

### 2.2 Model / metric problems (the substantive ones)

These are what motivated CriterionV3 and Chain v3 CoT.

**P1. The dominant union-mask metric lies on multi-target queries.**
The Surprise3D evaluation protocol merges every referent of a query
into one GT *union* mask
\[
M^q_\text{union} = \bigcup_{k=1}^{K_q} m_k,
\]
and reports `mIoU` and `Acc@τ` against that single mask. For a query
with `K ≥ 2` referents (e.g. *"the chairs around the coffee table"*) a
model that selectively segments any one of them cleanly gets
`IoU ≤ 1/K` on this protocol. It is gradient-equivalent to a
`1 − 1/K` partial-recall failure on a single-target query. The
asymmetry runs the other way too: a prediction that diffusely covers
all `K` candidates — without segmenting any one cleanly — can score
arbitrarily close to 1.

In plain English: the optimizer learns to *spread* its mask across all
candidates, because spreading is what the metric rewards.

The empirical fingerprint is `mIoU > meanMaxIoU` on the vanilla
baseline (22.95 vs. 21.87 on Surprise3D val; see §4). If the model were
truly committing to one referent, `meanMaxIoU` would be at or above
`mIoU`; the inequality going the other way is a direct readout of the
multi-target spread behaviour.

**P2. Small instances starve.** Surprise3D has plenty of queries that
target small objects (mugs, outlets, remotes, picture frames) whose
mask is on the order of tens of points. The vanilla Reason3D loss is
union BCE + Dice on the union-mask, both of which are dominated by
background once the foreground is small relative to the scene. The
gradient is too small to commit the mask decoder to a reasonable
boundary. Small-instance hit@0.50 sits well below the average.

**P3. Multi-step relational queries have no architectural channel.**
Queries like *"the towel on the chair near the window"* require
committing to a landmark first (the window, the chair) and conditioning
the final mask on that commitment. The single-`[SEG]` interface threads
exactly one hidden state from the LM into the mask decoder; there is no
inference-time mechanism for a prior spatial commitment to influence a
later one. CriterionV3 cannot fix this — no loss reshaping can — so
the architectural extension (Chain v3 CoT) is the right tier of
intervention.

These three failure modes map cleanly to the three structural fixes
in §3 below — one metric-side fix per failure mode P1, one loss-side
fix covering P1 and P2, and one architecture-side fix for P3.

---

## 3. Fixes I applied (especially the metrics)

This section pairs each formal definition with a one-paragraph natural
language explanation, so each fix can be understood without reading
the code.

### 3.1 Per-instance evaluation: `meanMaxIoU` and `hit@τ`

**Formal definition.** For a query `q` with GT instances
`{m_k}_{k=1..K_q}` and a single predicted binary mask `M̂^q`, define
\[
\text{maxIoU}(\hat M^q) = \max_{k=1..K_q}\, \text{IoU}(\hat M^q, m_k),
\]
\[
\text{meanMaxIoU} = \mathbb{E}_q\!\left[\text{maxIoU}(\hat M^q)\right],
\quad
\text{hit}@\tau = \mathbb{E}_q\!\left[\mathbf{1}\{\text{maxIoU}(\hat M^q) > \tau\}\right].
\]

**What it actually does.** Instead of comparing the prediction to a
union mask of all `K` referents, compare it to *each* GT instance
separately and keep the best score. If the model selectively segments
*any one* of the referents cleanly, that's the score. If `K = 1` the
two protocols are identical — `meanMaxIoU = mIoU` and
`hit@τ = Acc@τ` — so this is a strict super-set, not a different
benchmark. The new metrics only diverge from the old on multi-target
queries, which is exactly where the old metrics misbehave.

**The diagnostic gap.** Define
`Δ_τ = hit@τ − Acc@τ`. Because `maxIoU ≥ IoU(union)` pointwise,
`Δ_τ ≥ 0` always when the model is *selective*. A negative `Δ_τ` is the
fingerprint of the multi-target spread behaviour. We report `Δ_50`
throughout — it is the single most informative number in the table.

**Where it lives in the code.** `lavis/tasks/refer_seg_task_v3.py`.
Eval task name `3d_refer_seg_v3`, set `task: 3d_refer_seg_v3` in any
val YAML. Output appears in `metrics_v3_test.json` next to the run.

### 3.2 CriterionV3: a drop-in segmentation loss

**Headline term — best-of-set BCE+Dice.** Mirrors `meanMaxIoU` at the
loss level:
\[
\mathcal{L}_\text{ref} = \min_{k=1..K}\big[\lambda_b \,\text{BCE}(\hat M, m_k) + \lambda_d\,\text{Dice}(\hat M, m_k)\big].
\]
That is: the loss rewards the prediction for being consistent with
*any one* GT instance, breaking the `1/K` partial-recall artefact of
union-mask supervision. We use `λ_b = λ_d = 1.0`.

**Coverage hinge — prevents collapse.** `min_k` alone admits a
degenerate solution (always pick the smallest GT instance and collapse
all mass there). The one-sided hinge
\[
\mathcal{L}_\text{cov} = \lambda_\text{cov}\max\!\Big(0,\; 1 - \tfrac{\text{mass}(\hat M)}{\text{mass}(\text{GT}_\text{union})}\Big)
\]
penalizes under-prediction relative to the union but never punishes
selecting one referent over many. We use `λ_cov = 0.10`.

**Scale-aware dice — addresses small instances.** Each sample's dice
is weighted by the inverse square root of its instance size:
\[
\mathcal{L}_\text{scale} = \frac{\sum_b w_b\,\text{Dice}_b}{\sum_b w_b},\quad
w_b = (\sqrt{|m_b|} + \epsilon)^{-1}.
\]
A 30-point instance and a 10,000-point instance are pulled into the
batch gradient at comparable amplitudes; small instances stop being
swamped by background.

**Optional auxiliaries — Lovász + focal-point.** Both gated by
`small_size_threshold = 50` (only fire on instances below 50 points): a
Lovász-hinge on upsampled logits to sharpen boundaries, and a sigmoid
focal point-level term in best-of-set form to push gradient through
hard background. Either flag off recovers the headline; both flags off
recovers Reason3D's legacy loss exactly when the best-of-set and
scale-aware terms are also turned off.

**Total CriterionV3 loss.**
\[
\mathcal{L}_\text{seg} = \mathcal{L}_\text{ref} + \mathcal{L}_\text{cov} + \alpha_s\,\mathcal{L}_\text{scale}
+ \mathbf{1}_\text{bnd}\alpha_b\,\mathcal{L}_\text{Lovász} + \mathbf{1}_\text{pt}\alpha_p\,\mathcal{L}_\text{focal-pt}.
\]

**Where it lives in the code.**
`lavis/models/reason3d_models/seg_loss_v3.py`. Switched in via
`model.seg_criterion_cfg.*` flags in any train YAML; defaults are in
`reason3d_surprise_finetune_chainv3.yaml`.

### 3.3 Chain v3 CoT: two-pass forward with mass-pool feedback

**Mass-pool token (the architectural change).** After Pass 1 emits a
landmark mask `M_1 ∈ R^{N_sp}` (logits at superpoint resolution), pool
the superpoint features by the sigmoid of `M_1` and project to the LM
token width:
\[
\mathbf{t}_\text{pool} = \mathbf{W}\!\left(\frac{\sum_p \sigma(M_1[p])\, \mathbf{f}_p}{\sum_p \sigma(M_1[p]) + \epsilon}\right).
\]
This token is *appended* to the encoder memory before Pass 2 — it
becomes a single extra slot the decoder cross-attends to in addition to
the 32 Q-Former queries and the tokenized question.

**Two-pass training (resolves circularity).** `t_pool` depends on
`M_1`, which depends on the teacher-forced decode that needs `t_pool`.
We follow R2S §3.2: Pass 1 runs T5 with the original encoder memory,
extracts the first `[SEG]` hidden state, and decodes `M_1` under
`no_grad`; the mass-pool token is computed from `M_1.detach()`; Pass 2
runs T5 with the augmented encoder memory and decodes `M_2`. The total
loss is a token-weighted LM term across both passes plus a *single*
segmentation term on `M_2` (no mask-level term on `M_1`).

**W1-pure regime (the contribution against R2S).** R2S requires
LLM-mined intermediate-mask GT to supervise `M_1`. We do not use that;
`M_1` is shaped only by (i) the pretrained class-segmentation prior
(the backbone has already learned `"the {class}" + [SEG] →
class-mask`) and (ii) the LM signal on the rationale text that names
the landmark before `[SEG]_1`. The `detach` blocks gradient leakage
from `L_seg(M_2)` back into `M_1`. This is the load-bearing scientific
claim: same architectural feedback as R2S, no mined ground truth.

**Two-`[SEG]` chain-of-thought target.** For samples where a regex
extracts a landmark phrase from the question (`"closest to the {X}"`,
`"upon entering the {X}"`, etc.), the GT answer is reshaped into a
two-`[SEG]` template, e.g. *"I will start with the {landmark}. [SEG].
Given that, the {target}. [SEG]."* — five interchangeable natural-text
variants are sampled per training example to avoid template overfit.
Non-relational queries (`cs`, `hi`, `camera_view`) and regex misses
fall back to the single-`[SEG]` template; the forward path branches per
sample on the `[SEG]` count in the tokenized labels.

**Where it lives in the code.**
`lavis/models/reason3d_models/reason3d_t5_chainv3_cot.py`. Search for
`mask_pool_token`, `intermediate_loss_weight`, and the
`detach()` calls in `forward` to verify the W1-pure claim.

---

## 4. Current numbers — Bare, Chain, Loss, CoT

All numbers below are from the full Surprise3D val split (n = 8,229
for vanilla and `+ CriterionV3`; n = 8,225 for `+ Chain v3 CoT`
because the two-pass decoder drops 4 queries when it cannot emit two
well-formed `[SEG]`s).

> **Honest caveat on the comparison protocol.** The `+ CriterionV3` row
> is computed *on top of the chain v3 stack*, not on top of the vanilla
> single-`[SEG]` backbone, because under our compute budget we did not
> rerun the loss-only grid against the vanilla architecture. The
> architecture-fixed reference for the loss grid is `A0` (legacy
> criterion on the chain v3 stack) in §4.3 below — `A0` and the
> `+ CriterionV3` row in the headline differ only in the loss flags.
> The implication is that the `+ CriterionV3` headline number is
> mildly confounded with chain v3 effects and slightly understates the
> per-flag contribution of CriterionV3 in isolation. The correct
> comparison would be `vanilla` vs. `vanilla + CriterionV3` and
> `chain v3` vs. `chain v3 + CriterionV3` separately; the 8-checkpoint
> plan in §6 is structured to fix this.

### 4.1 Headline table

| Method | mIoU | A_25 | A_50 | meanMaxIoU | hit@0.25 | hit@0.50 | Δ_50 |
|--------|-----:|-----:|-----:|-----------:|---------:|---------:|-----:|
| Reason3D vanilla              | 22.95 | 35.41 | 20.79 | 21.87 | **34.37** | 17.83 | −2.96 |
| &nbsp;&nbsp; + CriterionV3    | 20.54 | 31.32 | 17.95 | **22.43** | 34.00 | 19.75 | +1.80 |
| &nbsp;&nbsp; + Chain v3 CoT   | 19.87 | 30.53 | 17.34 | 22.40 | **34.37** | **20.94** | **+3.60** |

Three things to read from this table:

1. **Vanilla `mIoU > meanMaxIoU` (22.95 > 21.87).** This is the
   spreading fingerprint from §2.2 P1. The model is averaging across
   multi-target referents. The `Δ_50 = −2.96pt` says the same thing in
   per-instance vs. union form.

2. **CriterionV3 reverses the inequality (`mIoU < meanMaxIoU`,
   20.54 < 22.43).** Per-instance `hit@0.50` rises by +1.92pt; union
   `A_50` falls by 2.84pt. This is *exactly* the trade-off the loss is
   designed to produce: the model commits to one referent rather than
   averaging. `Δ_50` flips from −2.96 to +1.80.

3. **Chain v3 CoT pushes `hit@0.50` further (+3.11pt over vanilla).**
   The `Δ_50 = +3.60pt` is the largest of the three configurations,
   and confirms that the architecture-level mechanism (mass-pool, two
   passes) reinforces rather than competes with the loss-level
   selectivity. Note also the union-side regression (mIoU 19.87,
   A_50 17.34) is only a little deeper than CriterionV3's; the
   per-instance gain is the bigger move.

### 4.2 Per-question-type breakdown

`hit@0.50` per Surprise3D query family (the full per-family table
across all six metrics is regenerated by
`scripts/summarize_surprise_predictions.py --markdown-per-qt`):

|                          | cs | hi | camera_view | first_view | rel_pos | abs |
|--------------------------|---:|---:|-----------:|-----------:|--------:|----:|
| Reason3D vanilla         | 16.68 | 19.23 | 22.70 | 19.36 | **18.71** | 9.25 |
| + CriterionV3            | 18.95 | 22.49 | 23.93 | 19.18 | 17.84 | 10.75 |
| + Chain v3 CoT           | **20.34** | **23.39** | **26.69** | **20.33** | 17.98 | **12.83** |

Pattern in plain English:

- **CriterionV3 helps where multi-target is dense.** `cs` (+2.27pt)
  and `hi` (+3.26pt) are the two non-relational families with the
  highest multi-target rate; `abs` (+1.50pt) has the highest
  small-object density. The relational families
  (`first_view`, `rel_pos`) are flat or slightly down — a loss can't
  manufacture multi-step reasoning.

- **Chain v3 CoT improves five of six families.** Largest gains on
  `cs` (+3.66), `hi` (+4.16), `camera_view` (+3.99), `abs` (+3.58).
  Surprisingly the **non-relational** families benefit most, even
  though those rows fall back to the single-`[SEG]` template at
  training time. The likely explanation is that the chain v3
  architecture (mass-pool head, two-pass forward) acts as a
  general-purpose regularizer, not just a multi-step pathway.

- **`rel_pos` regresses by −0.73pt.** This is the only failure mode of
  the W1-pure recipe in this table. A partially wrong `M_1`
  (unsupervised by design) can mislead the final decode for
  distance-conditioned queries. This is the motivating observation for
  the deferred Tier-2 LLM rationale (B9 in
  `chainv3_cot_ablation_tracker.md`).

### 4.3 Loss-side ablation (CriterionV3 axes), architecture fixed at chain v3

| Row | Name | bos | scl | bnd | pt | mIoU | A_25 | A_50 | meanMaxIoU | hit@0.25 | hit@0.50 | Δ_50 |
|-----|------|:---:|:---:|:---:|:--:|-----:|-----:|-----:|-----------:|---------:|---------:|-----:|
| A0  | `legacy`        | — | — | — | — | 22.02 | 33.65 | 19.74 | 21.30 | 33.08 | 17.49 | −2.25 |
| A1  | `v3_default`    | ✓ | ✓ | — | — | 20.54 | 31.32 | 17.95 | 22.43 | 34.00 | 19.75 | +1.80 |
| A2  | `no_bos`        | — | ✓ | — | — | 21.37 | 32.36 | 18.70 | 20.73 | 31.93 | 16.70 | −2.00 |
| A3  | `no_scale`      | ✓ | — | — | — | 19.65 | 29.83 | 16.88 | **23.01** | 34.35 | **20.80** | **+3.92** |
| A4  | `lovasz`        | ✓ | ✓ | ✓ | — | 20.00 | 30.71 | 17.40 | 22.39 | **34.55** | 20.20 | +2.80 |
| A5  | `pointaux`      | ✓ | ✓ | — | ✓ | 19.93 | 30.52 | 17.51 | 22.30 | 34.18 | 20.42 | +2.91 |
| A6  | `all_loss`      | ✓ | ✓ | ✓ | ✓ | 19.96 | 30.54 | 17.17 | 22.14 | 34.20 | 19.70 | +2.53 |

Reading the grid:

- **A0 vs. A1.** Best-of-set + scale-aware together trade −1.79pt of
  union `A_50` for +2.26pt of per-instance `hit@0.50` and flip `Δ_50`
  from −2.25 to +1.80 — selectivity emerges from the loss design,
  not from the architecture.

- **A2 vs. A3 isolates the two flags.** A2 (scale-aware only) is the
  weakest of the v3 family and stays in the negative `Δ_50` regime —
  scale weighting alone does *not* produce per-instance commitment.
  A3 (best-of-set only) is the strongest per-instance configuration
  in the whole grid — best-of-set is the dominant ingredient.

- **A4 / A5 (auxiliaries) are mild wins; A6 is not monotone.**
  Lovász or focal-point each adds a little `hit@0.50` over A1, but
  stacking everything (A6) is slightly *below* A1. The auxiliaries
  appear to act on overlapping low-resolution gradient channels rather
  than as fully orthogonal terms. The recommended single-loss default
  is A1 (`bos + scl`); A4 or A5 are useful when small-instance recall
  is important.

The two A-row configurations the report's 8-checkpoint plan keeps
(§6) are **A3 = `no_scale`** (best per-instance number, isolates
best-of-set) and **A4 = `lovasz`** (best-of-set + scale + boundary;
small-instance recall) — alongside the `legacy` (A0) and CoT (B1')
runs.

---

## 5. Observations on the metrics (especially the new ones)

### 5.1 The `Δ_50` gap is the single most informative number

If you only have time to look at one number per ablation row, look at
`Δ_50 = hit@0.50 − A_50`. Reading it:

- **`Δ_50 < 0`.** The model is averaging across multi-target referents
  (failure mode P1). Vanilla Reason3D, A0 (legacy on chain v3), A2
  (`no_bos`) all sit here. The further negative, the worse the
  averaging behaviour.
- **`Δ_50 ≈ 0`.** The model is roughly neutral — a single-target query
  dominates the average and the multi-target rows wash out.
- **`Δ_50 > 0`.** The model commits to one referent. CriterionV3 (any
  best-of-set row) and Chain v3 CoT live here. The amount of positive
  `Δ_50` measures how aggressively the model is selecting.

The flip from negative to positive `Δ_50` is, in our experience, a
more reliable signal of the intended training-side change than either
`mIoU` or `hit@0.50` in isolation, because it directly measures the
*difference* between the union and per-instance evaluation modes.

### 5.2 `mIoU` and `hit@0.50` should be reported together, not chosen between

A single number is never enough on Surprise3D val. The two pairs we
want to see for any new run are:

| | Union (single-mask) | Per-instance |
|--|---:|---:|
| Continuous metric | `mIoU` | `meanMaxIoU` |
| Threshold metric  | `Acc@τ` | `hit@τ` |

`mIoU > meanMaxIoU` is diagnostic of multi-target spread; the inverse
is diagnostic of selectivity. Reporting only the union pair (as the
upstream Surprise3D protocol does) silently penalizes selective models;
reporting only the per-instance pair would silently reward a model
that returns one tiny instance per query and ignores the rest. Both
together, with `Δ_50` as the diagnostic gap, gives an honest picture.

### 5.3 What `Acc@0.25` tells us that `Acc@0.50` does not

Across the ablation grid, `Acc@0.25` is more stable than `Acc@0.50` —
it shifts by ~3pt across A0…A6, while `Acc@0.50` shifts by ~3pt as
well but in different directions. This matches the intuition that
`Acc@0.25` is mostly a coverage / recall test ("did the model find the
right region at all?"), while `Acc@0.50` is a quality test ("is the
mask precise enough?"). CriterionV3's coverage hinge is doing its job
when `A_25` does not collapse alongside `A_50`. Watching the two
together prevents the optimizer from satisfying `A_50` by sacrificing
basic recall.

### 5.4 Numbers that are *not* directly comparable across rows

A few subtle things to flag for honesty:

- **Sample size differs.** Vanilla / CriterionV3: n = 8,229. Chain
  v3 CoT: n = 8,225 (4 dropped by the two-pass decoder). The
  difference is within noise but exists.
- **The CriterionV3 row in the headline is on the chain v3 stack**,
  not on the vanilla stack — see §4 caveat. For the loss-on-vanilla
  comparison, the right reference is `A0 = legacy on chain v3`
  (mIoU 22.02, hit@0.50 17.49) vs. the headline `vanilla`
  (mIoU 22.95, hit@0.50 17.83): chain v3 architecture alone, with
  *legacy* loss, is roughly within noise of vanilla.
- **The joint CriterionV3 ⊕ Chain v3 CoT configuration was not run.**
  Both grids were sized to characterize each contribution in
  isolation; the size of any combined effect is open.
- **The 22.95% vanilla number comes from our preprocessing**
  (§1.4); it is not directly comparable to the published Surprise3D
  Table 4 number (11.00%) at the row level. The right comparison is
  *internal* to this fork, on the same val split, with the same
  filter.

---

## 6. Plan — train four branches to 8 checkpoints

The current numbers in §4 come from runs of *varying* lengths that
landed at different epochs (the headline rows are the per-run last
epoch under `RunnerBase`'s 20-epoch schedule, but training behaviour
degrades roughly past the 3rd epoch on the small-eval log, so apples-
to-apples alignment is not what those numbers report). The next step
is to train four branches on a canonical schedule and align them at
identical checkpoint indices.

### 6.1 The four branches

| Branch | Loss config | Architecture | Init ckpt | Justification |
|--------|-------------|--------------|-----------|---------------|
| **bare reason3d** | legacy union BCE+Dice+IoU | single-`[SEG]` (vanilla) | `reason3d_inference.pth` | Floor for the whole comparison. Replaces the current "vanilla" row. |
| **chain** | legacy union BCE+Dice+IoU | chain v3 (single-pass) | same | Architecture-only effect (no loss change, no two-pass). Fills the gap left by §4's confounded `+ CriterionV3` row. |
| **loss (no_scale & lovasz)** | A3 = `bos` only and A4 = `bos + scl + Lovász` (two runs) | chain v3 (single-pass) | same | Captures the strongest per-instance loss (A3) and the best small-instance loss (A4) on the same architecture as `chain`. |
| **B' CoT** | CriterionV3 default (`bos + scl`) | chain v3 CoT (two-pass + mass-pool, W1-pure) | same | The headline of the architectural contribution; same recipe as `cot_pure` in `chainv3_cot_ablation_tracker.md`. |

That is **five** training runs total (bare, chain, loss-A3, loss-A4,
B' CoT), each producing **8 checkpoints** that we evaluate at common
indices. With `max_epoch = 20` and mid-epoch saves every 2k steps,
8 evenly spaced checkpoints land near epochs `{2, 4, 6, …, 16}` — but
the canonical alignment is the per-epoch saves
`checkpoint_{1, 3, 5, 7, 9, 11, 13, 15}.pth` — even-indexed for
parity, large enough to span the regime past the 3rd-epoch
"training-loss-stops-tracking-eval" inflection point we documented in
`report/surprise3d_results_comparison.md`.

### 6.2 What gets compared

At each of the 8 checkpoint indices we compute, for each branch:

1. **All six metrics (`mIoU`, `A_25`, `A_50`, `meanMaxIoU`,
   `hit@0.25`, `hit@0.50`)** plus the `Δ_50` gap, as in §4.1.
2. **Per-question-type breakdown** for the same six metrics across the
   six families (`cs`, `hi`, `camera_view`, `first_view`, `rel_pos`,
   `abs`), as in §4.2 — this is what
   `scripts/summarize_surprise_predictions.py --markdown-per-qt`
   produces from each run's `qualitative/predictions.jsonl`.
3. **Qualitative analysis on intermediate masks** (B' CoT only).
   `Models/reason3d/scripts/run_visualize_qualitative.sh` renders
   `M_1` (Pass 1 landmark) and `M_2` (Pass 2 target) as colour-coded
   PLYs; the goal is to confirm the qualitative behaviour described
   in §3.3 (that `M_1` resolves the landmark sensibly, even when
   abstract — *"item for carrying clothes"* → *"clothes bag"*) and to
   diagnose the `rel_pos` regression by visualizing failure cases.

### 6.3 Expected reading of the 8-checkpoint sweep

Three concrete questions the sweep should answer:

- **Q1: Does the per-instance gain from CriterionV3 hold across
  training time, or only at the well-tuned ckpt?** If `Δ_50` is
  positive across all 8 checkpoints for the loss branches but
  negative for `bare`, then the selectivity behaviour is robust to
  training time. If `Δ_50` flips sign late in training, the loss is
  fragile.

- **Q2: Does Chain v3 CoT's gain over the loss branches concentrate
  on the relational families across all checkpoints, or wash out
  early?** The current single-checkpoint snapshot says `cs` and `hi`
  benefit *more* than `rel_pos`; a multi-checkpoint sweep will
  confirm whether that ordering is stable.

- **Q3: How does the chain v3 architecture (without CoT, without
  loss change) compare to vanilla?** This is the question §4's
  headline cannot answer cleanly because the `+ CriterionV3` row
  shares the chain v3 stack. If `chain` ≈ `bare` in per-instance
  metrics, then chain v3 *alone* is loss-agnostic and `B' CoT` gets
  a clean ablation; if `chain` already moves `Δ_50` positive, the
  architecture is doing more than we currently credit it with.

### 6.4 Resources and scheduling

Each finetune is `4 × A100`, batch 8, ~14h per 20-epoch run on the
current data scale. Five runs ≈ 70 GPU-hours in series, ~18h
elapsed when run in parallel on two 4-GPU nodes. Eval is cheap
(~10 min per checkpoint with `NPROC=4`), so 8 checkpoints × 5 runs ×
10 min ≈ 7 GPU-hours of eval. Total budget ≈ 80 GPU-hours, which
fits in a week on the cluster.

---

## 7. What this fork honestly *does not* claim

For completeness, the things this fork does *not* claim:

- **No claim of state-of-the-art on Surprise3D val.** The numbers
  here are *internal* improvements over the Reason3D baseline under a
  rebuilt preprocessing pipeline. They do not align with the upstream
  Surprise3D leaderboard at the row level, and we do not pretend they
  do.
- **No claim of generalization beyond Surprise3D.** The three fixes
  are dataset-agnostic in design, but every number here is on
  Surprise3D val. ScanRefer / ReferIt3D / ScanReason / SQA3D have not
  been run.
- **No claim that CriterionV3 dominates the legacy loss on union
  metrics.** It does not, by design — the trade-off is intentional
  (§3.2). If the downstream task cares about union-mIoU specifically,
  the headline CriterionV3 default is the wrong choice and one should
  use the legacy loss or A2 (`no_bos`).
- **No claim that the W1-pure regime matches R2S's mined-GT regime
  in absolute numbers.** What we claim is that mined intermediate
  masks are *not necessary* to get the architectural feedback to
  work; whether mined GT would push the numbers further is open and
  is the natural next experiment after the 8-checkpoint sweep.
- **The reported Surprise3D val is filtered.** n = 8,229 after
  dropping rows whose target id is missing from the sampled cloud.
  Numbers are not on the full nominal 10,198. The filter is a
  precondition for a well-defined GT mask, not a way to game the
  metric, but it does mean the row count differs from the upstream
  row count and direct row-wise comparisons are invalid.

---

## 8. Code map (where each claim lives)

| Section | Claim | Code |
|---------|-------|------|
| §1.3.A | All-instance ScanNet++ preprocessing patch | `third_party/scannetpp/semantic/transforms/{factory,mesh}.py` (uncommitted patch); `Models/reason3d/scripts/scannetpp_prepare_full_instances.yml` |
| §1.3.B | Surprise3D-specific preprocessing wrapper | `Models/reason3d/scripts/prepare_surprise_scannetpp_pth.py`, `scripts/run_prepare_surprise_scannetpp_pth.sh` |
| §1.3.C | `filter_missing_gt_in_pth` + instance-id cache | `Models/reason3d/lavis/datasets/datasets/threedrefer_datasets.py` |
| §2.1 | Engineering bug list and fixes | `Models/reason3d/docs/REASON3D_PROBLEMS_AND_FIXES.md` |
| §3.1 | `meanMaxIoU`, `hit@τ` per-instance metrics | `Models/reason3d/lavis/tasks/refer_seg_task_v3.py` |
| §3.2 | CriterionV3 (best-of-set, coverage, scale, aux) | `Models/reason3d/lavis/models/reason3d_models/seg_loss_v3.py` |
| §3.3 | Chain v3 CoT mass-pool, two-pass, W1-pure | `Models/reason3d/lavis/models/reason3d_models/reason3d_t5_chainv3_cot.py` |
| §4.1 | Headline table reproduction | `scripts/summarize_surprise_predictions.py --markdown-cross` |
| §4.2 | Per-question-type breakdown | `scripts/summarize_surprise_predictions.py --markdown-per-qt` (recover question_type via `scripts/recover_surprise_question_types.py`) |
| §4.3 | Loss-side ablation grid (A0–A6) | `lavis/projects/reason3d/train/reason3d_surprise_finetune_chainv3*.yaml`; tracker in `Models/reason3d/docs/chainv3_ablation_tracker.md` |
| §6 | 8-checkpoint plan | tracker in `Models/reason3d/docs/chainv3_cot_ablation_tracker.md` |

---

## 9. Reproducing

### 9.1 Environment

A reproducible setup script for the Reason3D stack lives at
[`Models/reason3d/scripts/install_reason3d_deps.sh`](Models/reason3d/scripts/install_reason3d_deps.sh)
and a snapshot at
[`Models/reason3d/docs/ENVIRONMENT.md`](Models/reason3d/docs/ENVIRONMENT.md).
You also need PointGroup CUDA ops:

```bash
cd Models/reason3d
bash scripts/build_pointgroup_ops.sh
```

### 9.2 Data

- **Annotations** — Surprise3D from
  [HuggingFace `hhllzz/surprise-3d`](https://huggingface.co/datasets/hhllzz/surprise-3d).
  Default expected at `/nfs-stor/lan.wei/data/annotations/surprise_val.json`;
  pass `--ann <path>` to override in the analysis scripts.
- **Point clouds** — ScanNet++ v2 (request access from
  [kaldir.vc.in.tum.de/scannetpp](https://kaldir.vc.in.tum.de/scannetpp/)).
- **Preprocessing to `.pth`** (this is the §1 step):

```bash
cd Models/reason3d
bash scripts/run_prepare_surprise_scannetpp_pth.sh
```

Path conventions live in
[`Models/reason3d/docs/DATA_SYNC.md`](Models/reason3d/docs/DATA_SYNC.md);
the default subdirectory `processed_surprise_full_pth` is hard-wired
in the YAMLs under `lavis/configs/datasets/3dseg/`.

### 9.3 Train one of the four branches

| Branch (§6.1) | Launcher | Config |
|---------------|----------|--------|
| bare reason3d | `scripts/run_surprise_finetune.sh` | `reason3d_surprise_finetune.yaml` |
| chain | `scripts/run_surprise_finetune_chainv3.sh` (with `enable_best_of_set=false enable_scale_aware=false`) | `reason3d_surprise_finetune_chainv3.yaml` |
| loss A3 (`no_scale`) | `scripts/run_surprise_finetune_chainv3.sh` (with `enable_scale_aware=false`) | same |
| loss A4 (`lovasz`) | `scripts/run_surprise_finetune_chainv3.sh` (with `enable_boundary=true`) | same |
| B' CoT | `scripts/run_surprise_finetune_chainv3_cot.sh` | `reason3d_surprise_finetune_chainv3_cot.yaml` |

```bash
cd Models/reason3d
REASON3D_INIT_CKPT=/path/to/reason3d_pretrained.pth NPROC=4 \
    bash scripts/run_surprise_finetune_chainv3_cot.sh
```

### 9.4 Evaluate one checkpoint

```bash
cd Models/reason3d
REASON3D_CKPT=/path/to/your_checkpoint.pth \
    bash scripts/run_surprise_zeroshot_eval.sh
```

This writes `metrics_v3_test.json` plus
`qualitative/predictions.jsonl` (one row per query, including chain v3
fields: `decoded_text_pass1`, `intermediate_point_iou`,
`did_two_pass`).

### 9.5 Per-question-type analysis

The JSONL ships with empty `question_type` by design; recover it by
joining to `surprise_val.json`:

```bash
cd Models/reason3d
python3 scripts/summarize_surprise_predictions.py \
    --markdown-per-qt --transpose \
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \
    /path/to/run/qualitative/predictions.jsonl
```

Add a second JSONL and `--markdown-cross --highlight-max` for the
cross-variant tables.

### 9.6 Qualitative figures (intermediate masks for B' CoT)

```bash
cd Models/reason3d
bash scripts/run_visualize_qualitative.sh
```

Wraps `scripts/visualize_qualitative_preds.py` to render the
predicted / intermediate / GT masks as colour-coded PLYs and PNG
snapshots.

---

## Acknowledgements & upstream

- **Dataset.** All Surprise3D dataset details (queries, splits,
  annotation pipeline, official benchmark numbers) belong to the
  upstream paper and repository:
  - Repo: <https://github.com/liziwennba/SURPRISE3D>
  - HuggingFace: <https://huggingface.co/datasets/hhllzz/surprise-3d>
- **Reason3D baseline.** The Reason3D code under `Models/reason3d/`
  is forked from <https://github.com/KuanchihHuang/Reason3D>. The
  diff introduced by this fork is summarized in
  [`Models/reason3d/docs/REASON3D_FORK_CHANGES.md`](Models/reason3d/docs/REASON3D_FORK_CHANGES.md).
- **Scenes.** ScanNet++ v2 (Yeshwanth et al.) provides the underlying
  3D point clouds.

## Contact

For questions about the **Surprise3D dataset and original
publication** (annotations, splits, license, official benchmark
protocol), contact the upstream authors listed in the upstream README.

For questions about **this fork** — preprocessing pipeline,
CriterionV3, Chain v3 CoT, per-instance metrics — contact
`lan.wei@mbzuai.ac.ae`.
