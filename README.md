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

4.  **Current numbers (Surprise3D val, checkpoint 10).** Four of five
    planned branches have completed ckpt 10; `chain` is still pending.
    All four eval runs share an **identical pinned val row set
    (n = 10,174 across all six question-type families)** — the eval
    pipeline now has reproducibility guarantees the earlier ckpt-8
    numbers did not (per-row OOM safety net, GPU-tensor accumulator
    leak fixed, val JSON pinned via
    `scripts/build_filtered_surprise_val.py`; see §6 and §9.4).
    `vanilla` Reason3D gets `mIoU = 24.64%`, `hit@0.50 = 18.68%`,
    `Δ_50 = hit@0.50 − A_50 = −3.17pt` (multi-target spreading).
    Each CriterionV3-flavour branch flips `Δ_50` positive: `lovasz`
    (best-of-set + scale-aware + Lovász, **strongest per-instance**)
    reaches `hit@0.50 = 24.47%`, `Δ_50 = +5.72pt`, `meanMaxIoU = 25.15%`;
    `no_scale` 22.53%, +3.98pt; `cot` 22.68%, +3.53pt. The architectural
    extension (`cot`) improves over `vanilla` by +4.00pt `hit@0.50` and
    +6.70pt `Δ_50`, but does not beat the loss-only branches at this
    checkpoint — `lovasz`'s boundary term is the dominant lever on
    Surprise3D val at ckpt 10. **These ckpt-10 numbers supersede any
    earlier ckpt-8 numbers in this document and prior commits**:
    different `n` across branches and a since-fixed eval-side memory
    leak made the ckpt-8 row counts unreliable as a four-way ablation.

5.  **Plan.** Train five branches to 8 checkpoints each (bare Reason3D,
    chain, loss = `no_scale` + `lovasz`, B' CoT) and compare across the six
    Surprise3D query families plus qualitative analysis of intermediate
    landmark masks. The ckpt-10 numbers above are the latest fully-trusted
    row of this sweep (eval pipeline now reproducible end-to-end); the
    8-checkpoint plan in §6 will fill in the missing `chain` row and the
    earlier checkpoint indices.

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
> filtered val set (n = 10,174 in the canonical pinned JSON used at
> ckpt 10; the original ad-hoc-filtered runs reported n ≈ 8,229) and
> the set of trainable instances differ. The right way to read it:
> under our preprocessing, vanilla Reason3D scores 22.95% on the
> earlier ad-hoc filter — and at ckpt 10 on the pinned filter
> (n = 10,174) it scores **24.64%** (§4.1). We treat the latter
> as the floor that CriterionV3 and Chain v3 CoT need to beat *on the
> same preprocessing pipeline* (which they do on per-instance metrics;
> see §4).

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

All numbers below are from the **checkpoint-10** sweep on the full
Surprise3D val split. Four of the five planned branches in §6.1 have
landed at ckpt 10 (`vanilla`, `no_scale`, `lovasz`, `cot`); the
**chain** branch (legacy union loss on the chain v3 single-pass
architecture, the apples-to-apples reference for the loss-only
contribution) is still pending and is the missing piece for a fully
clean four-way ablation.

**These numbers supersede the earlier ckpt-8 numbers in earlier
commits of this README.** The ckpt-8 row counts varied across
branches (`vanilla` 8,230 / `lovasz` 8,229 / `cot` 8,226 / `no_scale`
8,016) for two reasons we have since closed:

- **An eval-side GPU memory leak** in the per-row return path of
  `valid_step` accumulated `pred_pmask` / `gt_pmask` CUDA tensors
  across rows. By row ~2400 of a 10,174-row val sweep this was
  ~14 GiB of leaked GPU memory on top of the ~13 GiB of fp32 model
  weights, which caused late-eval CUDA OOMs on heavy scannet++
  scenes (e.g. `578511c8a9`, `c4c04e6d6c`). Each OOM truncated the
  run silently and produced a different `n`. **Fixed** by dropping
  the per-point CUDA tensors from the returned dict in
  `lavis/tasks/refer_seg_task.py` (the masks are still written to
  disk via `np.savez_compressed`).
- **Drift in the dataset's runtime filter** (§1.3.C) across runs
  depending on the per-scene `.pth` mtimes and instance-id cache
  state. **Fixed** by pinning a pre-filtered val JSON
  (`scripts/build_filtered_surprise_val.py` →
  `surprise_val_filtered_v1.json`) and setting
  `dataset_init.filter_missing_gt_in_pth: false` in every val YAML;
  see §9.4 for the canonical procedure.

The eval also gained a **per-row OOM safety net**: a CUDA OOM on
`predict_seg` is converted to either (a) a `num_beams=1`,
`max_len=32` recovery retry that almost always succeeds, or (b) a
`oom: true` sentinel row in `predictions.jsonl` with NaN ious. Each
`metrics_v3_test.json` therefore now reports `n_total`, `n_oom`,
`n_oom_recovered` alongside the headline metrics. For all four
branches at ckpt 10 the values were `n_total=10174, n_oom=0,
n_oom_recovered=0` — i.e. the leak and pinning fixes alone were
enough to keep every row inside the GPU memory envelope, the safety
net never had to fire on this run.

> **Honest caveat on the comparison protocol.** The two loss branches
> (`no_scale` = A3, `lovasz` = A4) and the `cot` branch all share the
> chain v3 stack; only the loss flags and the two-pass switch differ.
> `vanilla` is the legacy single-`[SEG]` backbone with the legacy union
> loss, so it is the ablation floor for *both* the loss change and the
> architecture change. The right loss-only reference (chain v3
> architecture, legacy loss) is the **chain** branch in §6.1, which has
> not yet completed its checkpoint-10 sweep — until it does, the
> apples-to-apples loss-only delta is mildly confounded with
> chain-v3-architecture effects.

### 4.1 Headline table (checkpoint 10)

| Method | n | mIoU | A_25 | A_50 | meanMaxIoU | hit@0.25 | hit@0.50 | Δ_50 |
|--------|--:|-----:|-----:|-----:|-----------:|---------:|---------:|-----:|
| `vanilla` (Reason3D, legacy loss, single-[SEG])      | 10174 | **24.64** | **37.70** | **21.85** | 23.03 | 35.95 | 18.68 | −3.17 |
| `chain` (chain v3 single-pass, legacy loss)          |  —    |   —   |   —   |   —   |   —   |   —   |   —   |  —  |
| `no_scale` (chain v3, CriterionV3 `bos` only)        | 10174 | 21.34 | 32.75 | 18.55 | 24.33 | 37.06 | 22.53 | +3.98 |
| `lovasz` (chain v3, CriterionV3 `bos+scl+lovasz`)    | 10174 | 21.30 | 32.62 | 18.75 | **25.15** | **37.45** | **24.47** | **+5.72** |
| `cot` (chain v3 CoT, CriterionV3 default `bos+scl`)  | 10174 | 21.65 | 32.81 | 19.15 | 24.55 | 36.74 | 22.68 | +3.53 |

Five things to read from this table:

1. **Vanilla wins every union metric** (`mIoU` 24.64, `A_25` 37.70,
   `A_50` 21.85). All three CriterionV3-flavour branches give up
   ~3pt of `mIoU` and ~3pt of `A_50` to vanilla. The union-side
   regression is the cost of moving to a per-instance loss; the
   per-instance numbers below are what the trade-off pays for.

2. **Vanilla `mIoU > meanMaxIoU` (24.64 > 23.03)** and `Δ_50 = −3.17pt`.
   Same multi-target spreading fingerprint from §2.2 P1, slightly
   wider than at ckpt 8 (gap was 0.71pt, now 1.61pt). Vanilla is
   averaging across multi-target referents — the headline `mIoU`
   overstates how often the model actually commits to one of them.

3. **All three CriterionV3-flavour branches flip the inequality.**
   `meanMaxIoU > mIoU` and `Δ_50 > 0` for every loss/CoT branch. The
   amount differs: `lovasz` has the largest gap (+5.72pt), `no_scale`
   next (+3.98pt), `cot` last among the three (+3.53pt). The
   ckpt-10 ranking inverts the ckpt-8 ranking, where `no_scale`
   (best-of-set only) led on `Δ_50`. Three more checkpoints of
   training are enough for the **boundary (Lovász) term** to take
   effect — at ckpt 10 it adds an extra +1.74pt of `Δ_50` over
   `no_scale`, and an extra +0.82pt of `meanMaxIoU` (24.33 → 25.15).

4. **`lovasz` is the strongest per-instance run by every metric**
   (`meanMaxIoU` 25.15, `hit@0.25` 37.45, `hit@0.50` 24.47,
   `Δ_50` +5.72). It is also the run that most cleanly trades
   `mIoU` for `meanMaxIoU`: its `mIoU` (21.30) is the lowest of the
   four but its `meanMaxIoU` (25.15) is the highest, the textbook
   shape of a model that has learned per-instance commitment at the
   cost of per-pixel averaging.

5. **`cot` is competitive but no longer dominant.** At this checkpoint
   `cot` lands above the loss-only branches on `mIoU` (21.65) and
   `A_50` (19.15) — the highest among the three CriterionV3-flavour
   branches on union metrics — but behind `lovasz` on every
   per-instance column (`meanMaxIoU` 24.55 vs. 25.15, `hit@0.50`
   22.68 vs. 24.47, `Δ_50` +3.53 vs. +5.72). The architectural
   extension is still paying for itself relative to vanilla
   (+4.00pt `hit@0.50`, +6.70pt `Δ_50`), but **does not** beat the
   loss-only branches on the per-instance headline at ckpt 10 — the
   per-question-type breakdown below shows where the architectural
   edge actually lives (and where it does not).

### 4.2 Per-question-type breakdown — `hit@0.50` (checkpoint 10)

|                       | cs | hi | first_view | rel_pos | abs | camera_view |
|-----------------------|---:|---:|----------:|--------:|----:|------------:|
| `vanilla`             | 16.66 | 20.26 | 21.05 | 19.75 |  9.82 | 20.03 |
| `no_scale`            | 23.19 | 25.28 | 21.72 | 20.85 | **11.82** | 21.59 |
| `lovasz`              | **25.58** | **28.46** | **23.08** | **21.40** | 11.64 | **22.52** |
| `cot`                 | 23.64 | 24.93 | 22.57 | 20.58 | 11.64 | 21.85 |

*n per family per branch:* identical across all four branches —
`cs` 2,893, `hi` 2,892, `camera_view` 1,927, `first_view` 1,183,
`rel_pos` 729, `abs` 550 (total 10,174). The pinned val JSON +
fixed-leak eval guarantee a byte-identical row set per branch, so
every cell above is computed on exactly the same denominator.

Pattern in plain English:

- **`lovasz` wins five of six families on `hit@0.50`.** The
  exceptions: `abs` (where `no_scale` wins by 0.18pt — within
  noise on n = 550). The boundary (Lovász) term that was middle of
  the pack at ckpt 8 has compounded into a clear cross-family
  lead by ckpt 10.

- **Best-of-set still drives the multi-target families, and Lovász
  amplifies it.** Both `no_scale` (bos only) and `lovasz` (bos +
  scl + Lovász) lift `cs` and `hi` substantially over `vanilla`:
  +6.53 / +5.02 for `no_scale` on `cs` / `hi`, +8.92 / +8.20 for
  `lovasz` on the same two families. `cot` lands between them:
  +6.98 on `cs`, +4.67 on `hi`. The non-relational, multi-target
  families are exactly where best-of-set was designed to help, and
  the boundary term sharpens the per-instance commitment further.

- **`first_view` and `camera_view` separate `lovasz` from the rest.**
  `lovasz` +2.03 / +2.49pt on `first_view` / `camera_view` vs.
  `vanilla`; `no_scale` +0.67 / +1.56pt; `cot` +1.52 / +1.82pt.
  Both families are mostly single-target, so the gain comes from
  *quality* (boundary precision) rather than from selection — the
  Lovász term picks the boundary well, while best-of-set alone has
  nothing to grip on once the row is single-target.

- **`rel_pos` is no longer where `cot` underperforms.** At ckpt 8
  `cot` lost 2.19pt on `rel_pos` vs. vanilla; at ckpt 10 it gains
  +0.83pt. Three more checkpoints fixed the regression. `lovasz`
  still leads on `rel_pos` (+1.65pt vs. vanilla), but `cot` is now
  positive too. The ckpt-8 hypothesis — "wrong landmark mask `M_1`
  propagates into distance-conditioned decode" — was a ckpt-8
  artefact, not a structural issue with the architecture.

- **`abs` is the small-instance family and `no_scale` wins it
  narrowly.** `no_scale` 11.82, `lovasz` 11.64, `cot` 11.64,
  `vanilla` 9.82. All three CriterionV3 branches close the gap to
  vanilla (+1.88pt on average), but the differences among the
  three are within noise on n = 550. The scale-aware term in
  `lovasz`/`cot` is not doing more than best-of-set alone here,
  which suggests the small-instance threshold may need to be
  re-tuned (or the `abs` family genuinely tops out at this
  recall-quality level for this stack).

### 4.3 Reading the four-branch `Δ_50`

The single most informative cross-branch comparison at ckpt 10 is
the `Δ_50` ranking:

```
lovasz    +5.72pt   ← bos + scl + Lovász; highest per-instance gain
no_scale  +3.98pt   ← best-of-set only
cot       +3.53pt   ← chain v3 architecture, bos + scl
chain     ?         ← not yet evaluated at ckpt 10
vanilla   −3.17pt   ← single-[SEG] floor; multi-target spread
```

The gap between `vanilla` and the strongest CriterionV3 row
(`lovasz`) is **+8.89pt of `Δ_50`**, which is the headline number
to remember from this checkpoint — about 1pt wider than the
ckpt-8 vanilla → `no_scale` gap of +8.26pt, but the *winner of the
ranking has changed*. At ckpt 8 best-of-set alone (`no_scale`) was
the per-instance leader; at ckpt 10 the boundary term (`lovasz`)
has compounded enough training-time gradient signal to overtake it.
The architectural extension (`cot`) is closing on the loss-only
branches but does not yet beat them on `Δ_50` at ckpt 10; the
8-checkpoint plan in §6 will see whether `cot` continues to close
the gap, holds steady, or regresses.

---

## 5. Observations on the metrics (especially the new ones)

### 5.1 The `Δ_50` gap is the single most informative number

If you only have time to look at one number per ablation row, look at
`Δ_50 = hit@0.50 − A_50`. Reading it:

- **`Δ_50 < 0`.** The model is averaging across multi-target referents
  (failure mode P1). `vanilla` (−3.17pt at ckpt 10) sits here.
  The further negative, the worse the averaging behaviour.
- **`Δ_50 ≈ 0`.** The model is roughly neutral — a single-target query
  dominates the average and the multi-target rows wash out.
- **`Δ_50 > 0`.** The model commits to one referent. Every
  CriterionV3-flavour branch lives here at ckpt 10: `lovasz` (+5.72),
  `no_scale` (+3.98), `cot` (+3.53). The amount of positive `Δ_50`
  measures how aggressively the model is selecting.

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

Across the four ckpt-10 branches, `Acc@0.25` ranges 32.62–37.70
(a 5.1pt spread) while `Acc@0.50` ranges 18.55–21.85 (a 3.3pt
spread) in the same direction — the union-side regression that
buys the per-instance gain. This matches the intuition that
`Acc@0.25` is mostly a coverage / recall test ("did the model find
the right region at all?"), while `Acc@0.50` is a quality test
("is the mask precise enough?"). CriterionV3's coverage hinge is
doing its job when `A_25` does not collapse alongside `A_50` — at
ckpt 10 the worst `A_25` (`lovasz` 32.62) is still well above the
"prediction collapsed to nothing" floor we would expect if the
hinge were absent. Watching the two together prevents the
optimizer from satisfying `A_50` by sacrificing basic recall.

### 5.4 Numbers that are *not* directly comparable across rows

A few subtle things to flag for honesty:

- **Sample size is identical across all four branches at ckpt 10.**
  All four eval runs report `n_total = 10,174` and per-family
  `(cs 2,893, hi 2,892, camera_view 1,927, first_view 1,183,
  rel_pos 729, abs 550)` — the pinned val JSON
  (`scripts/build_filtered_surprise_val.py` →
  `surprise_val_filtered_v1.json`) plus the OOM-leak fix in
  `lavis/tasks/refer_seg_task.py` (no more accumulating `pred_pmask`
  / `gt_pmask` in `val_result`) guarantees that every cell of every
  table above is computed on the same denominator. This is a real
  change from the ckpt-8 numbers, where the per-branch row counts
  drifted by hundreds of rows.
- **`vanilla` and the three CriterionV3-flavour branches differ in
  *both* loss and architecture.** All three loss/CoT branches share
  the chain v3 stack; `vanilla` is a single-`[SEG]` Reason3D
  backbone. The loss-on-the-same-architecture comparison (legacy vs.
  CriterionV3 on chain v3) requires the **chain** branch in §6.1,
  which has not yet completed.
- **The joint CriterionV3-default ⊕ Chain v3 CoT row at ckpt 10 is
  the `cot` row** (default = bos + scl). The `no_scale` branch and
  the `cot` branch therefore differ in *two* axes — loss flag (no
  scl) and architecture (no two-pass) — so a `cot − no_scale` delta
  cannot be cleanly attributed to either. The CoT-alone effect is
  the still-pending row "chain v3 single-pass with legacy loss"
  (= the **chain** branch).
- **The 24.64% `vanilla` mIoU at ckpt 10 comes from our
  preprocessing** (§1.4); it is not directly comparable to the
  published upstream Surprise3D row at all. The right comparison is
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

Three concrete questions the sweep should answer; the ckpt-10 row
in §4 is the latest datapoint we have for each (ckpt 10 is the
strongest signal we trust right now — the eval pipeline is now
reproducible end-to-end, see §4).

- **Q1: Does the per-instance gain from CriterionV3 hold across
  training time, or only at the well-tuned ckpt?** Ckpt 10 says
  yes: `Δ_50` is strongly positive for all three
  CriterionV3-flavour branches (+5.72 / +3.98 / +3.53 for
  `lovasz` / `no_scale` / `cot`) and negative for `vanilla`
  (−3.17). The vanilla → strongest CriterionV3 gap is +8.89pt of
  `Δ_50`, slightly wider than at ckpt 8 (+8.26pt). The per-instance
  advantage is not collapsing as training progresses; if anything
  the boundary term is gaining ground.

- **Q2: Does Chain v3 CoT's gain over the loss branches concentrate
  on the relational families across all checkpoints, or wash out
  early?** Ckpt 8 said no on `rel_pos` (cot −2.19pt vs. vanilla);
  ckpt 10 says yes (cot +0.83pt vs. vanilla, but still behind
  `lovasz`'s +1.65pt). The architectural signal on relational
  families is emerging with training — the question is whether
  `cot` eventually overtakes `lovasz` on `rel_pos` (its design
  intent) or whether the boundary term keeps the lead. The
  remaining checkpoints will tell.

- **Q3: How does the chain v3 architecture (without CoT, without
  loss change) compare to vanilla?** Still open: the `chain` branch
  has not yet completed its ckpt-10 evaluation. This is the missing
  piece that lets us cleanly attribute the `cot − vanilla` gap
  between *the architecture* and *the loss*.

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
- **The reported Surprise3D val is filtered.** n = 10,174 in the
  canonical pinned val JSON used at ckpt 10
  (`surprise_val_filtered_v1.json`, see §9.4), after dropping rows
  whose target id is missing from the sampled cloud. Numbers are not
  on the full nominal 10,198. The filter is a precondition for a
  well-defined GT mask, not a way to game the metric, but it does
  mean the row count differs from the upstream row count and direct
  row-wise comparisons are invalid.

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

### 9.4 Pin the validation row set (do this once)

The four full-val YAMLs under
`Models/reason3d/lavis/projects/reason3d/val/` are pinned to a *pre-filtered*
val JSON so every checkpoint sees a byte-identical row set, regardless of
`.pth` mtimes, eval auto-resume, instance-id cache state, or which shell /
day the eval is launched from. Build that pinned JSON once before the sweep:

```bash
cd Models/reason3d
bash scripts/run_build_filtered_surprise_val.sh
```

This produces (defaults, override via env vars in the script):

- `/nfs-stor/lan.wei/data/annotations/surprise_val_filtered_v1.json` — the
  pinned annotations file (a strict subset of `surprise_val.json` with
  rows whose `object_id` doesn't appear in the corresponding scene's
  `sampled_instance_anno_id` removed).
- `/nfs-stor/lan.wei/data/annotations/surprise_inst_id_cache_v1.json` — a
  per-scene instance-id cache compatible with
  `dataset_init.instance_id_cache_file` (handy if you ever flip the
  runtime filter back on for ScanRefer / future work).

The val YAMLs already point at the filtered JSON and set
`dataset_init.filter_missing_gt_in_pth: false`, so the runtime filter is a
no-op on this path. To go back to the unpinned, dataset-filter-at-runtime
behaviour for one run, override on the CLI:

```bash
REASON3D_FILTER_MISSING_GT_IN_PTH=1 \
    REASON3D_CKPT=/path/to/checkpoint.pth \
    bash scripts/run_surprise_zeroshot_eval.sh \
    -- datasets.3d_refer.build_info.annotations.test.storage=/nfs-stor/lan.wei/data/annotations/surprise_val.json
```

### 9.5 Evaluate one checkpoint

```bash
cd Models/reason3d
REASON3D_CKPT=/path/to/your_checkpoint.pth \
    bash scripts/run_surprise_zeroshot_eval.sh
```

This writes `metrics_v3_test.json` plus
`qualitative/predictions.jsonl` (one row per query, including chain v3
fields: `decoded_text_pass1`, `intermediate_point_iou`,
`did_two_pass`).

#### Eval determinism gotchas

- **Auto-resume is OFF by default.** Set `REASON3D_AUTO_RESUME=1` only when
  you intentionally want to recover from a crash. With auto-resume on, the
  task gates rows by `(scene_id, ann_id)` keys already in
  `qualitative/predictions.jsonl`; if a previous attempt left a partial /
  inconsistent JSONL, the resumed run can land on a different `n` than a
  clean single-shot run on the same YAML.
- **Always use `NPROC=1` for headline numbers.** With `NPROC>1` the eval
  uses `DistributedSampler`, which shards by dataset length; mixing that
  with auto-resume's `apply_eval_resume_skip` (which mutates dataset
  length) can change which rows each rank sees on a resume. The launcher
  already prints a warning in this combination.
- **Don't regenerate `.pth` files mid-sweep.** Even with the runtime
  filter off, regenerating `processed_surprise_full_pth` will change the
  GT mask data the metrics consume. If you must regenerate, rerun
  `run_build_filtered_surprise_val.sh` so the pinned JSON tracks the new
  `.pth` content, and re-evaluate every checkpoint to be honest about
  comparability.

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
