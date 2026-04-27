# Related tasks: **models** and benchmarks (one-line reviews)

Each line is: **name** — *input* → *output*; *core mechanism / how it is usually evaluated*.

Sources: Surprise3D (Huang et al., NeurIPS 2025) experiments and Table 1; standard 3D-VL literature for models not named on every Surprise3D page.

---

## I. Models explicitly used as Surprise3D baselines

(Surprise3D appendix: five systems with adapted ScanNet++ preprocessing.)

| Model | One-line |
|-------|----------|
| **MLLMfor3D** | Point cloud + text prompt → segmentation-style 3D outputs; **label-free** pipeline that lifts **2D pseudo-masks** (or image-derived cues) into 3D, used as a segmentation-capable MLLM baseline on Surprise3D (mIoU / Acc@τ). |
| **3D-Vista** | 3D scene representation + language → **masked language–masked 3D alignment** training; on Surprise3D the released setup is adapted to predict **bounding boxes** (with detector head / UniDet3D-style processing), evaluated with **accuracy**, not mask IoU. |
| **Reason3D** | Point cloud + sentence → **instance mask** + short LM decode ending in `[SEG]`; **Q-Former** bridges sparse 3D encoder (e.g. SPFormer-style) to **Flan-T5**, then a **hierarchical mask decoder** reads the `[SEG]` hidden state—**the segmentation line your work extends**. |
| **Intent3D** | Scan-style 3D + **implicit intention** language → **object boxes** (detection / grounding); **LLM-generated or intention-aware prompts** drive a detector—Surprise3D evaluates it as a **bbox** baseline (accuracy), not dense mask. |
| **ChatScene** | 3D scene + dialogue or instruction → **language + grounding** using **learned object ID tokens** as a bridge; Surprise3D adapts the released model toward **per-instance segmentation predictions** plus dialogue-style training (e.g. also uses **SQA3D** in pretraining per paper). |

**Supporting backbone (not a “language benchmark” but a model):** **UniDet3D** — multi-dataset **3D detector** producing proposals/boxes; Surprise3D uses it to generate boxes for bbox-centric baselines and for **superpoint** / preprocessing compatible with Reason3D-style pipelines.

---

## II. Other **segmentation / mask–language** models (same task family as Reason3D)

**Why Surprise3D did not report numbers on every model in this section**

The paper evaluates **five** systems (§I) and states that **official code had to be adapted to ScanNet++** without a turnkey recipe—so practical constraints dominate, not “these methods are irrelevant.”

| Factor | What it implies for §II models |
|--------|--------------------------------|
| **Engineering budget** | Each baseline needs preprocessing (superpoints, labels, text format), training/finetuning, and a fair eval protocol on a **new** benchmark; a dataset paper cannot port every concurrent mask+LM system. |
| **Paradigm coverage** | They already include **two mask-output lines** (**MLLMfor3D**, **Reason3D**) plus **three** non-mask or hybrid MLLM/detection/dialogue lines. **SegPoint** is in the same broad family as **Reason3D** (LLM → mask); adding SegPoint is often **redundant** unless the goal is a full leaderboard sweep. |
| **Release / compatibility** | **Grounded 3D-LLM**, **MORE3D**, etc. may lack a maintained path for **ScanNet++** point clouds, the same superpoint graph, or the same instance mask definition Surprise3D uses—without that, numbers are not comparable. |
| **Publication timing** | **Ning et al. (ICCV 2025; 3D ReasonSeg + R2S)** and some **Reasoning3D** lines can be **concurrent or later** than the Surprise3D experiment freeze; authors may not have had stable code or a clear duty to reproduce every overlapping arXiv. |
| **Task mismatch** | **Reasoning3D**-style work often targets **part-level** or different mask semantics; mapping to Surprise3D’s **whole-instance union mask** over ScanNet++ instances is **non-trivial** and may not be what those papers report out of the box. |

So: §II lists **close relatives** for related-work and **future** leaderboard expansion; absence from Surprise3D tables is expected for a **first benchmark release**, not a claim that those methods are weaker or out of scope.

| Model | One-line |
|-------|----------|
| **SegPoint** | Point cloud + free-form instruction → **segmentation**; **LLM-driven** pipeline that maps language to point-level or instance masks (Surprise3D Table 1 cites it as mask + LLM language source). |
| **MORE3D** | **MLLM** + 3D reasoning aimed at **joint reasoning and segmentation-class outputs**; cited in Surprise3D §2 alongside Reason3D/SegPoint as LLM-heavy 3D segmentation directions. |
| **Grounded 3D-LLM** | Builds on the **3D-LLM** line with explicit **grounding** so language aligns to **3D masks**; template/LLM-generated training pairs (Table 1: mask output). |
| **Ning et al., ICCV 2025** — *Enhancing Spatial Reasoning in Multimodal Large Language Models through Reasoning-based Segmentation* | [arXiv:2506.23120](https://arxiv.org/abs/2506.23120) · [OpenAccess PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ning_Enhancing_Spatial_Reasoning_in_Multimodal_Large_Language_Models_through_Reasoning-based_ICCV_2025_paper.pdf). **3D ReasonSeg** is the **dataset name inside this paper** (not the paper title)—point cloud + complex instruction → **mask** (25k train / 4k val). If search engines do not find “3D ReasonSeg”, search the **full paper title** or arXiv id. |
| **R2S (same paper)** | **Relevant Reasoning Segmentation**: two-stage framework—(1) identify **relevant** elements, (2) **prior-guided refinement** for the final mask; paired with **3D ReasonSeg** for training/eval. |
| **Reasoning3D** (2025 line, “reasoning in 3D” / part reasoning) | Large **VLM** + 3D → **fine-grained or part-level** reasoning segmentation; emphasizes **parts** and open-vocabulary reasoning rather than whole-object Surprise3D-style referents. |

---

## III. **MLLM / LLM–3D** models (language + boxes or dialogue; usually not mask-IoU on Surprise3D)

| Model | One-line |
|-------|----------|
| **3D-LLM** | Discretized **3D scene tokens** (or anchors) injected into an **LLM** → **open-ended text** and coarse **3D references** (e.g. boxes); multi-task training on 3D captioning/QA/grounding-style data. |
| **LL3DA** | **Interactive** 3D instruction tuning → model outputs **language + 3D localization** (boxes / referents) in a dialogue; bridges **user–scene** interaction rather than single-shot mask IoU. |
| **LEO** | **Generalist** 3D embodied **foundation** model (world modeling + language) → planning/QA/grounding-style behaviors over 3D; breadth over a single referring-mask benchmark. |
| **Chat-3D** | **Object-centric IDs** or structured scene tokens tied to an **LLM** → **grounded chat** about specific instances; reduces coreference ambiguity vs. raw free text on points. |
| **ChatScene** | See **§I** (Surprise3D baseline): **ID-tokens** + scene graph–like linking for **dialogue + grounding** on ScanNet-class data. |
| **ScanReason — ReGround3D** | Implicit query → **bbox + reasoning text**; **interleaved “chain-of-grounding”** steps between an **MLLM reasoning module** and a **3D grounding module** (reasoning and localization alternate at inference). |

---

## IV. **Classic 3D referring expression grounding** (bbox on ScanNet / colored scans)

These are **models**, usually trained on **ScanRefer** and/or **ReferIt3D (Nr3D, Sr3D)**; **input** = point cloud (often with RGB) + sentence; **output** = **one (or few) 3D bounding boxes**; **method** = match sentence to object proposals then classify/regress.

| Model / family | One-line |
|----------------|----------|
| **ScanRefer (original ECCV method)** | **Two-branch** encoder over **point-cloud instances** and **sentence**, then **fusion + classification** to pick the referred object box—establishes the listener paradigm for 3D referring. |
| **EDA, TGNN, BUTD-style 3D detectors** | Early **graph- or detection-centric** encoders (object relation networks, TGNN, bottom-up–top-down **DETR**-style heads) **fuse** language with **3D object proposals** → bbox; strong on **named** referents, weak when object names are stripped (as Surprise3D’s diagnostic experiment shows). |
| **Transformer matchers (e.g. InstanceRefer, 3DVG-Transformer, MVT, SAT, ViewRefer, HAM)** | **Cross-modal transformers** between **word tokens** and **point / object / multi-view tokens** with **contrastive** or **hierarchical alignment** → bbox; SOTA line on ScanRefer/Nr3D/Sr3D until MLLMs absorbed the task. |

*(Exact roster changes year to year; use this block as “representative methods” on slides.)*

---

## V. **ScanReason** (benchmark + model pair)

| Item | One-line |
|------|----------|
| **ScanReason (dataset)** | ScanNet + **implicit / reasoning** questions → **answer + 3D location (bbox)**; five reasoning types (spatial, functional, logical, emotional, safety). |
| **ReGround3D (method)** | **MLLM** produces reasoning steps; **3D module** predicts boxes; **chain-of-grounding** interleaves the two—**reasoning-first**, bbox output (not the same as mask-IoU 3D-SRS). |

---

## VI. Benchmarks (datasets / tasks) — Surprise3D Table 1

Same content as before: **language-only QA/caption**, **bbox grounding**, **lang + bbox**, **mask datasets**, **Surprise3D**. See table below (unchanged substance).

### Language-only outputs (QA, captioning — no dense mask)

| Benchmark | One-line |
|-----------|----------|
| **CLEVR3D** | Synthetic 3D + templated questions → short answer; compositional logic, not real rooms. |
| **Scan2Cap** | ScanNet + target region → caption text; human dense captions. |
| **ScanQA** | ScanNet + question → short answer; 3D VQA. |
| **3DVQA** | 3D/multiview + templated Q → answer. |
| **SQA3D** | Situation / embodied QA on ScanNet → answer text. |
| **ScanScribe** | Scene → description text; template/LLM. |
| **3DMV-VQA** | Multiview 3D + Q → answer. |
| **M3DBench** | Broad 3D multimodal tasks → language; LLM prompts. |
| **SceneVerse** | Large-scale 3D–language → text; scale over mask grounding. |
| **MSQA** | Multistep 3D Q → answer; human + LLM. |
| **VLA-3D** | Templated 3D VLA-style Q → answer / instruction text. |
| **ExCap3D** | Multi-granularity captions → expressive text. |

### Bbox referring / grounding

| Benchmark | One-line |
|-----------|----------|
| **ReferIt3D** | Colored scan + utterance (Nr3D / Sr3D) → bbox of referred instance(s). |
| **ScanRefer** | ScanNet + sentence (usually names object) → bbox. |
| **3D-DenseOG** | Phrase → bbox per phrase. |
| **ScanEnts3D** | ScanRefer + phrase–entity links → bbox. |
| **PhraseRefer** | Phrase-level 3D referring → bbox. |
| **EmbodiedScan** | Embodied tasks + language → bbox-oriented labels. |
| **ScanReason** | Implicit reasoning Q → bbox (+ reasoning in pipeline). |
| **Intent3D** | Intention language → bbox. |

### Joint language + coarse geometry

| Benchmark | One-line |
|-----------|----------|
| **3D-LLM** | Scene tokens + instruction → text + coarse 3D refs. |
| **LL3DA** | Dialogue + 3D → bbox + language. |
| **3DMIT** | Instructions → language + bbox-style grounding. |
| **3D-GRAND** | Million-scale 3D–language → answers + bbox-level grounding. |

### Mask supervision (dataset / task definition)

| Benchmark | One-line |
|-----------|----------|
| **Grounded 3D-LLM** | Scene + text → mask (dataset/task in 3D-LLM line). |
| **SegPoint** | Instruction → mask (LLM-driven). |
| **Reason3D** | Query → mask + LM (referring-style training). |
| **Surprise3D** | Name-free query + ScanNet++ → **union mask**; 3D-SRS metrics. |

---

## VII. Other benchmarks **not** in Surprise3D Table 1

| Name | One-line |
|------|----------|
| **3D ReasonSeg** (dataset in Ning et al., ICCV 2025) | Reasoning language + point cloud → mask; see **§II** for paper title and arXiv. |
| **3DSRBench** | 3D spatial **VQA** → text (12 types; FlipEval). |
| **Anywhere3D-Bench** | Multi-level referring (object / part / space / activity) → bbox-like grounding. |
| **Open3D-VQA** | Aerial 3D VQA → text. |
| **Nr3D / Sr3D** | ReferIt3D splits → bbox. |

---

## VIII. One-slide positioning (models + benchmarks)

- **Surprise3D evaluates both:** mask specialists (**Reason3D**, **MLLMfor3D**) and **dialogue/box** MLLMs (**3D-Vista**, **Intent3D**, **ChatScene**) under one name-free protocol—shows **mask** models are still the right abstraction for dense spatial reasoning segmentation.  
- **Your stack:** same **Reason3D** I/O as above; **chain** changes only **what the LM is trained to say** before `[SEG]`, improving **auditability** of the same mask head.  
- **For “related models” talks:** group as (1) **bbox listeners** on ScanRefer, (2) **MLLM 3D agents** (3D-LLM, LL3DA, Chat-Scene, ScanReason), (3) **mask+LM** (SegPoint, Reason3D, MLLMfor3D, Grounded 3D-LLM, Ning et al. / 3D ReasonSeg).

If you want this split into two files (`related_models_one_liners.md` vs `related_benchmarks_one_liners.md`), say the word.
