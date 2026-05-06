<div align="center">

# Chain-of-Thought Segmentation for Name-Free Multi-Reference 3D Queries

*Course-project fork of [SURPRISE3D](https://github.com/liziwennba/SURPRISE3D) ·
Reason3D baseline + three structural fixes for name-free 3D referring
segmentation*

**Lan Wei** &nbsp;·&nbsp; `lan.wei@mbzuai.ac.ae` &nbsp;·&nbsp; MBZUAI &nbsp;·&nbsp; Draft v2 — May 2026

[![Report](https://img.shields.io/badge/Report-main__v2.tex-orange)](report/main_v2.tex)
[![Upstream dataset](https://img.shields.io/badge/Upstream-SURPRISE3D-blue)](https://github.com/liziwennba/SURPRISE3D)
[![Reason3D baseline](https://img.shields.io/badge/Baseline-Reason3D-green)](https://github.com/KuanchihHuang/Reason3D)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What this repository is

This is a **final-course-project fork** of the public
[SURPRISE3D](https://github.com/liziwennba/SURPRISE3D) release (NeurIPS 2025
Datasets & Benchmarks Track). The dataset, project page, and benchmark
citation belong to the upstream repository and paper — see
[Acknowledgements](#acknowledgements--upstream).

The **engineering and writing in this fork** are mine. The contribution is a
stack of three orthogonal structural fixes on top of the
[Reason3D](https://github.com/KuanchihHuang/Reason3D) baseline for
**name-free 3D Spatial Reasoning Segmentation (3D-SRS)** on Surprise3D, plus
the full LaTeX report that explains why each fix exists, what failure mode it
targets, and what it buys empirically.

---

## TL;DR — three structural fixes

The dominant single-`[SEG]` recipe for LM-driven 3D referring segmentation
collapses three structurally distinct sub-problems — **selecting matching
instance(s)**, **supervising small instances**, and **resolving multi-step
relational structure** — into one bottleneck. Each fix targets one of those.

1. **Per-instance evaluation protocol** — `meanMaxIoU` and hit$@\tau$.
   Scores selective multi-target predictions correctly and exposes the
   `mIoU > meanMaxIoU` gap that aggregated union-IoU hides.
   *Code:* [`Models/reason3d/lavis/tasks/refer_seg_task_v3.py`](Models/reason3d/lavis/tasks/refer_seg_task_v3.py)

2. **CriterionV3** — drop-in segmentation criterion with best-of-set
   BCE+Dice, coverage hinge, size-normalized dice, and optional
   Lovász-boundary + sigmoid-focal auxiliaries on a small-instance subset.
   Architecture-preserving, replaces the Reason3D loss only.
   *Code:* [`Models/reason3d/lavis/models/reason3d_models/seg_loss_v3.py`](Models/reason3d/lavis/models/reason3d_models/seg_loss_v3.py)

3. **Chain v3 CoT** — multi-step architectural extension. The LM emits an
   intermediate landmark `[SEG]`, the mass-pooled landmark superpoint
   feature is appended to the encoder memory, and a second `[SEG]` decodes
   the final target mask. Unlike the closest precedent (R2S), Chain v3 needs
   **no LLM-mined intermediate-mask GT** — the landmark mask is shaped only
   by the pretrained class-segmentation prior and the chain-of-thought
   language-modelling signal (a **W1-pure** regime).
   *Code:* [`Models/reason3d/lavis/models/reason3d_models/reason3d_t5_chainv3_cot.py`](Models/reason3d/lavis/models/reason3d_models/reason3d_t5_chainv3_cot.py)

---

## Headline numbers (Surprise3D val)

Numbers reproduced from `report/main_v2.tex` Table 2 (full table including
per-question-type breakdown lives in §6 / Appendix A of the report). All
percentages.

| Method | mIoU | A$_{25}$ | A$_{50}$ | meanMaxIoU | hit$@0.25$ | hit$@0.50$ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Reason3D (vanilla)               | 22.95 | 35.41 | 20.79 | 21.87 | **34.37** | 17.83 |
| &nbsp;&nbsp; + CriterionV3       | 20.54 | 31.32 | 17.95 | **22.43** | 34.00 | 19.75 |
| &nbsp;&nbsp; + Chain v3 CoT (W1-pure) | 19.87 | 30.53 | 17.34 | 22.40 | **34.37** | **20.94** |

- Vanilla `mIoU > meanMaxIoU` (22.95 > 21.87) is the signature of a model
  that *spreads* its prediction across multi-target referents instead of
  committing to one.
- CriterionV3 reverses this direction (20.54 < 22.43); per-instance
  hit$@0.50$ rises +1.92pt while union $A_{50}$ falls −2.84pt — the
  intentional per-instance ↔ union trade-off.
- Chain v3 CoT pushes hit$@0.50$ to **20.94** (+3.11pt over vanilla) and
  flips the per-instance gap $\Delta_{50}\!=\!\textsc{hit}@0.50\!-\!A_{50}$
  from −2.96pt on vanilla to **+3.60pt**.

The joint `CriterionV3 ⊕ Chain v3 CoT` configuration was not run under the
compute budget and is documented as deferred future work in the report's
limitations section.

---

## Repository layout

```
SURPRISE3D/
├── report/                                  # CVPR-style report (compile main_v2.tex)
│   ├── main_v2.tex                          # ← the deliverable
│   ├── main.bib  preamble.tex  cvpr.sty
│   ├── fig/                                 # teaser, chain v3 arch, qualitative panels
│   ├── surprise3d_results_comparison.md     # cross-run comparison
│   ├── surprise100_attribution_analysis.md  # 100-sample error attribution
│   └── surprise100_error_attribution.csv
│
├── Models/
│   ├── reason3d/                            # ← the active codebase for this fork
│   │   ├── lavis/
│   │   │   ├── models/reason3d_models/
│   │   │   │   ├── reason3d_t5.py                # vanilla baseline
│   │   │   │   ├── reason3d_t5_chainv3.py        # chain v3 architecture (single-pass)
│   │   │   │   ├── reason3d_t5_chainv3_cot.py    # ← Chain v3 CoT (two-pass + mass-pool)
│   │   │   │   ├── seg_loss.py                   # legacy loss
│   │   │   │   └── seg_loss_v3.py                # ← CriterionV3
│   │   │   ├── tasks/
│   │   │   │   ├── refer_seg_task.py             # union-IoU only
│   │   │   │   └── refer_seg_task_v3.py          # ← per-instance metrics
│   │   │   └── projects/reason3d/train/
│   │   │       ├── reason3d_surprise_finetune.yaml
│   │   │       ├── reason3d_surprise_finetune_v2.yaml
│   │   │       ├── reason3d_surprise_finetune_chainv3.yaml
│   │   │       └── reason3d_surprise_finetune_chainv3_cot.yaml
│   │   ├── scripts/
│   │   │   ├── run_surprise_finetune*.sh         # finetune launchers
│   │   │   ├── run_surprise_zeroshot_eval*.sh    # eval launchers
│   │   │   ├── summarize_surprise_predictions.py # per-question-type analysis
│   │   │   ├── recover_surprise_question_types.py# join predictions ↔ surprise_val.json
│   │   │   ├── visualize_qualitative_preds.py    # PLY / mask visualization
│   │   │   └── architecture_reason3d_*.py        # architecture diagrams
│   │   └── docs/                                  # design-space + ablation notes
│   │       ├── chainv3_cot_design_space.md
│   │       ├── chainv3_cot_ablation_tracker.md
│   │       ├── chainv3_cot_literature_review.md
│   │       ├── finetune_eval_scripts.md          # script-by-script reference
│   │       ├── REASON3D_FORK_CHANGES.md
│   │       └── REASON3D_PROBLEMS_AND_FIXES.md
│   └── intent3d/                            # second baseline (Intent3D, less central)
│
├── third_party/                             # scannetpp, unidet3d (vendored)
└── README.md                                # this file
```

For a script-by-script breakdown of every finetune / eval entry point,
required env vars, and YAML default, read
[`Models/reason3d/docs/finetune_eval_scripts.md`](Models/reason3d/docs/finetune_eval_scripts.md).

---

## Reproducing the report

### 0. Environment

A reproducible setup script for the Reason3D stack lives at
[`Models/reason3d/scripts/install_reason3d_deps.sh`](Models/reason3d/scripts/install_reason3d_deps.sh)
and a reference snapshot at
[`Models/reason3d/docs/ENVIRONMENT.md`](Models/reason3d/docs/ENVIRONMENT.md).
You'll also need PointGroup CUDA ops:

```bash
cd Models/reason3d
bash scripts/build_pointgroup_ops.sh
```

### 1. Data

- **Annotations** — Surprise3D (NeurIPS 2025) from
  [HuggingFace `hhllzz/surprise-3d`](https://huggingface.co/datasets/hhllzz/surprise-3d).
  Default expected at `/nfs-stor/lan.wei/data/annotations/surprise_val.json`
  by the analysis scripts; pass `--ann <path>` to override.
- **Point clouds** — ScanNet++ v2 (request access from
  [kaldir.vc.in.tum.de/scannetpp](https://kaldir.vc.in.tum.de/scannetpp/)).
- **Preprocessing to `.pth`** — Reason3D consumes per-scene `.pth` files
  with superpoints. Use:
  ```bash
  cd Models/reason3d
  bash scripts/run_prepare_surprise_scannetpp_pth.sh
  ```
  Path conventions live in
  [`Models/reason3d/docs/DATA_SYNC.md`](Models/reason3d/docs/DATA_SYNC.md);
  the default subdirectory `processed_surprise_full_pth` is hard-wired in
  the YAMLs under `lavis/configs/datasets/3dseg/`.

### 2. Train

Each row in the headline table corresponds to one finetune launcher. Set
`REASON3D_INIT_CKPT` to a Reason3D `.pth` checkpoint and (optionally)
`NPROC` for multi-GPU.

| Row in Table 2 | Launcher | Config |
|:--|:--|:--|
| Reason3D (vanilla) | `scripts/run_surprise_finetune.sh` | `reason3d_surprise_finetune.yaml` |
| + CriterionV3 (on chain v3 stack) | `scripts/run_surprise_finetune_chainv3.sh` | `reason3d_surprise_finetune_chainv3.yaml` |
| + Chain v3 CoT (W1-pure) | `scripts/run_surprise_finetune_chainv3_cot.sh` | `reason3d_surprise_finetune_chainv3_cot.yaml` |

```bash
cd Models/reason3d
REASON3D_INIT_CKPT=/path/to/reason3d_pretrained.pth NPROC=4 \
    bash scripts/run_surprise_finetune_chainv3_cot.sh
```

### 3. Evaluate (zero-shot or from a checkpoint)

```bash
cd Models/reason3d
REASON3D_CKPT=/path/to/your_checkpoint.pth \
    bash scripts/run_surprise_zeroshot_eval.sh
```

This writes `metrics_v3_test.json` plus
`qualitative/predictions.jsonl` (one row per query, including chain v3
fields: `decoded_text_pass1`, `intermediate_point_iou`, `did_two_pass`).

### 4. Per-question-type analysis

The JSONL ships with empty `question_type` by design; recover it by joining
to `surprise_val.json`:

```bash
cd Models/reason3d
python3 scripts/summarize_surprise_predictions.py \
    --markdown-per-qt --transpose \
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \
    /path/to/run/qualitative/predictions.jsonl
```

This prints the per-family table in §6.2 and Appendix A of the report
(rows = metric, columns = `cs / hi / first_view / relative_position / abs /
camera_view / TOTAL`). Add a second JSONL and `--markdown-cross
--highlight-max` to produce the cross-variant ablation tables.

### 5. Qualitative figures

```bash
cd Models/reason3d
bash scripts/run_visualize_qualitative.sh
```

Wraps `scripts/visualize_qualitative_preds.py` to render the predicted /
intermediate / GT masks as colour-coded PLYs and PNG snapshots. The
qualitative panels in `report/fig/` (`qual_relational.png`,
`qual_small.png`) were produced from the chain v3 CoT run at
`reason3d_surprise_zeroshot_chainv3_cot/20260506110758/`.

### 6. Compile the report

```bash
cd report
pdflatex main_v2 && bibtex main_v2 && pdflatex main_v2 && pdflatex main_v2
```

`main_v2.tex` uses CVPR's `cvpr.sty` (vendored locally) and the
`figincl{file}{caption}` helper that falls back to a labelled placeholder
box if a figure file is missing — so the document compiles cleanly
mid-edit even when `fig/*.png` is incomplete.

---

## Where each report claim lives in the code

| Report § | Claim | Code |
|:--|:--|:--|
| §3 | Per-instance metrics (`meanMaxIoU`, hit$@\tau$) | `lavis/tasks/refer_seg_task_v3.py` |
| §4.1 | Failure mode 1: union-mask spread | `seg_loss.py` (legacy union BCE+Dice) |
| §4.2 | CriterionV3: best-of-set + coverage hinge + sized dice | `lavis/models/reason3d_models/seg_loss_v3.py` |
| §4.3 | Chain v3 CoT mass-pool feedback, two-pass forward | `lavis/models/reason3d_models/reason3d_t5_chainv3_cot.py` |
| §4.3 | W1-pure regime (no $M_1$ supervision) | search for `intermediate_loss_weight` in the chainv3 CoT model |
| §6.1 | Main results table | `scripts/summarize_surprise_predictions.py` (--markdown-cross) |
| §6.2 + App. A | Per-question-type breakdown | `scripts/summarize_surprise_predictions.py --markdown-per-qt` |
| App. A | Question-type recovery from `surprise_val.json` | `scripts/recover_surprise_question_types.py` (uses `scripts/surprise_pred_join.py`) |
| App. B | Loss-grid ablations (A0…A6) | `lavis/projects/reason3d/train/reason3d_surprise_finetune_chainv3*.yaml` |
| App. C | Chain template variants (B1…B9) | `Models/reason3d/docs/chainv3_cot_design_space.md` |

---

## Notes / caveats

- **Sample sizes.** Vanilla and CriterionV3 rows are evaluated on
  $n\!=\!8{,}229$ Surprise val queries; the Chain v3 CoT row is on
  $n\!=\!8{,}225$ — the two-pass decoder drops 4 queries when it cannot
  emit two well-formed `[SEG]`s. This is documented in the caption of
  Table 2.
- **The CriterionV3 row in Table 2 is on the chain v3 stack**, not on
  vanilla Reason3D. The architecture-fixed reference (A0 = legacy
  criterion on the same chain v3 stack) is in Appendix B's loss-grid
  table; see Limitations in §7 of the report.
- **`report/surprise3d_results_comparison.md`**, the older small-eval
  cross-run summary, was the artefact accompanying the course submission
  alongside the report PDF. The numbers in the README and report tables
  are from the full $n\!\sim\!8{,}229$ val split, not the small subset.

---

## Acknowledgements & upstream

- **Dataset.** All Surprise3D dataset details (queries, splits,
  annotation pipeline, official benchmark numbers) belong to the upstream
  paper and repository:
  - Paper: [arXiv:2507.07781](https://arxiv.org/abs/2507.07781)
  - Repo: <https://github.com/liziwennba/SURPRISE3D>
  - HuggingFace: <https://huggingface.co/datasets/hhllzz/surprise-3d>
- **Reason3D baseline.** The Reason3D code under `Models/reason3d/` is
  forked from <https://github.com/KuanchihHuang/Reason3D>. The diff
  introduced by this fork is summarized in
  [`Models/reason3d/docs/REASON3D_FORK_CHANGES.md`](Models/reason3d/docs/REASON3D_FORK_CHANGES.md).
- **Scenes.** ScanNet++ v2 (Yeshwanth et al.) provides the underlying
  3D point clouds.

The companion 2D method MLLM-For3D (Huang et al., NeurIPS 2025) is
unrelated to this fork; see <https://github.com/tmllab/2025_NeurIPS_MLLM-For3D>
if you need the 2D-MLLM-adaptation angle.

---

## Citing

If anything from this fork is useful, please **cite Surprise3D first** —
without the upstream dataset and benchmark there is nothing to do:

```bibtex
@inproceedings{huang2025surprise3d,
  title     = {SURPRISE3D: A Dataset for Spatial Understanding and
               Reasoning in Complex 3D Scenes},
  author    = {Huang, Jiaxin and Li, Ziwen and Zhang, Hanlue and
               Chen, Runnan and Gao, Zhengqing and He, Xiao and
               Guo, Yandong and Wang, Wenping and Liu, Tongliang
               and Gong, Mingming},
  booktitle = {Advances in Neural Information Processing Systems
               (NeurIPS), Datasets and Benchmarks Track},
  year      = {2025}
}
```
---

## Contact

For questions about the **Surprise3D dataset and original publication**
(annotations, splits, license, official benchmark protocol), contact the
upstream authors listed in the upstream README.

For questions about **this fork** — CriterionV3, Chain v3 CoT,
per-instance metrics, the report — contact `lan.wei@mbzuai.ac.ae`.
