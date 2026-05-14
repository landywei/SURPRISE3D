# Multimodal Chain-of-Thought for 3D Inputs (Point Clouds)

> One-line pitch: a 3D model takes a point cloud as input — the geometric information is **complete** — yet today's 3D-LLMs underuse it because their CoT is text-only and shallow. Build a **multimodal CoT** where each reasoning step is itself a 3D-grounded artifact (sub-cloud, view, region, relation), so the model actually exploits the rich geometry it already has.

> **Independent of the harness idea.** This document is about *what a reasoning step looks like* inside a 3D model, not about inference-time scaffolding. The two ideas can be combined later but should be developed and evaluated separately.

---

## 1. Motivation

- Point clouds carry **complete 3D structure** (precise positions, distances, angles, occupancy).
- Intuition: with such rich input, a model should be able to do strong CoT spatial reasoning.
- Empirical reality (Du et al., *"The Point, the Vision and the Text"*, arXiv:2504.04540):
  - Pure LLMs *without* point cloud input outperform 3D-LLMs by **>10%** on 3D QA.
  - 3D-LLMs allocate far more attention to text tokens than to point cloud tokens.
  - Only a small subset of the point cloud actually contributes to the answer.
  - Average accuracy on binary spatial relations: ~5%.
- Diagnosis: the bottleneck is not the input — it is **how reasoning steps consume the input**. Text-only CoT cannot reference 3D structure precisely enough.

---

## 2. Hypothesis

> If reasoning steps themselves are **multimodal** (referencing concrete 3D primitives such as sub-clouds, 3D boxes, rendered views, or scene-graph fragments), the model is forced to ground its reasoning in the point cloud, closing the gap between *information available* and *information used*.

---

## 3. What "Multimodal CoT" Means for Point Clouds

A reasoning trajectory is a sequence of steps. Each step can be one of:

- **Reference-grounded text step**: text annotated with explicit pointers — object IDs, 3D bounding boxes, point indices, segment IDs.
- **Sub-cloud crop step**: model selects a 3D region of interest (box / sphere / mask); system re-encodes only that sub-cloud as the next observation.
- **Rendered view step**: model picks a camera pose; system renders a 2D image from the point cloud; image is fed back in (Think3D-style).
- **Scene-graph fragment step**: model emits a partial scene graph (objects + relations + boxes) as an intermediate symbolic artifact.
- **Qualitative-relation step**: model asserts qualitative relations (`above`, `left_of`, `inside`) on specific entity pairs; a checker validates them.
- **Latent 3D imagination step**: model predicts a latent 3D state (object pose, layout) as an intermediate (3DThinker-style).

Each step shape gives the model a different "vocabulary" to think with, all anchored in the point cloud.

---

## 4. Landscape (state of CoT for 3D)

### A. Textual CoT over 3D input
- **PointCoT** (arXiv:2602.23945) — Look-Think-Answer paradigm; *Point-Reason-Instruct* (~86K).
- **SceneCOT** ([scenecot.github.io](https://scenecot.github.io/)) — grounded 3D-scene CoT; 185K dataset.
- **3D-R1** (arXiv:2507.23478) — RL with perception + similarity + format rewards; Scene-30K dataset.
- **3D-CoT Benchmark** (arXiv:2503.06232) — hierarchical CoT annotations.

### B. Multimodal / grounded CoT (visual artifacts as steps)
- **Visual Sketchpad** ([visualsketchpad.github.io](https://visualsketchpad.github.io/)) — sketches as visual CoT (mostly 2D, conceptually relevant).
- **DeepSeek visual primitives** — points + boxes as grounded reasoning anchors.
- **Visual Reasoning Tracer** (arXiv:2512.05091) — object-level grounded reasoning paths.
- **SpatialThinker** (arXiv:2511.07403) — structured `<scene_graph>` + `<think>` + `<answer>`; multimodal trace.

### C. Latent / imagined 3D CoT
- **3DThinker** (arXiv:2510.18632) — 3D latent imagination during reasoning.
- **SpatialReasoner** (arXiv:2504.20024) — explicit 3D intermediate representation.

### D. Active / interactive 3D CoT
- **Think3D** (arXiv:2601.13029) — manipulates reconstructed point clouds via camera and view switching; +7.8% on BLINK Multi-view & MindCube. Has Think3D-RL variant.

### E. Diagnostic / evaluation
- **The Point, the Vision and the Text** (arXiv:2504.04540) — shows current 3D-LLMs underuse point clouds; introduces ScanReQA (forward/backward relations).
- Benchmarks: **SQA3D**, **ScanQA**, **ScanReQA**, **VSI-Bench**, **BLINK Multi-View**, **MindCube**.

---

## 5. Gap Analysis

| Work | Step type | Native point cloud input | Multimodal CoT steps | Verifies steps |
| --- | --- | --- | --- | --- |
| PointCoT | text | yes | partial (look step) | no |
| SceneCOT | text + grounding | partial | partial | no |
| 3D-R1 | text + rewards | yes | no | reward only |
| Think3D | text + view manipulation | reconstructed | yes (view) | no |
| 3DThinker | latent | yes | latent only | no |
| SpatialThinker | text + scene graph | RGB (not point cloud) | yes | implicit via reward |

**Unclaimed land**: a 3D model that natively ingests point clouds and produces CoT whose **every step is a 3D-grounded artifact** (sub-cloud, view, region, scene-graph fragment, or qualitative relation), with each artifact **independently verifiable** against the input geometry.

---

## 6. Concrete Proposal Variants

### (V1) Sub-Cloud CoT
At each step, the model emits a 3D region (box or mask). The system re-encodes that sub-cloud and provides it as the next observation. Final answer must reference selected sub-clouds.

### (V2) View-Sampling CoT
Like Think3D, but the input is a native point cloud (not reconstructed). Model picks camera poses; system renders; rendered images join the reasoning trace.

### (V3) Scene-Graph Step CoT
Reasoning trace contains intermediate scene-graph fragments as JSON (objects + relations + 3D boxes), then text reasoning, then answer. Borrow SpatialThinker's structured trace, but anchored in point cloud not RGB.

### (V4) Qualitative-Relation CoT with Verification
Each text step asserts qualitative relations on identified entities. A QCN consistency checker validates assertions and feeds back contradictions (or filters trajectories at training time).

### (V5) Hybrid Multimodal Trace
A trajectory may interleave any of the above step types. Model learns when to use which.

---

## 7. Minimum Viable Experiment

- Base 3D-LLM: PointLLM-V2 or 3D-LLM (whichever is strongest with code available).
- Benchmark: **SQA3D** (high human ceiling, low SOTA — clear headroom) + ScanReQA as a relations-focused probe.
- Pipeline: V3 (Scene-Graph Step CoT) as MVP because:
  - it's the cleanest single innovation,
  - inherits SpatialThinker-style structured trace which is known to work,
  - directly testable with simple format + answer rewards,
  - no need for a renderer or RL.
- Data: generate or use existing CoT annotations (PointCoT's Point-Reason-Instruct, SceneCOT-185K, Scene-30K).
- Training: SFT first; optional GRPO afterwards with format + accuracy + scene-graph-consistency rewards.
- Compare: vanilla 3D-LLM, PointCoT-style text-only CoT, ours.

---

## 8. Why This Could Be a Paper

- New thesis: **multimodal CoT is what unlocks point cloud input**, not more data or bigger models.
- Direct response to the "point cloud doesn't help" finding from ScanReQA paper.
- Strong baselines: PointCoT, SceneCOT, Think3D, 3D-R1.
- Clean ablation: textual CoT vs scene-graph CoT vs sub-cloud CoT vs hybrid.
- Generalization claim: structured 3D CoT transfers across 3D benchmarks (ScanQA / SQA3D / ScanReQA / VSI-Bench).

---

## 9. Risks / Failure Modes

- Generating high-quality 3D-grounded CoT data is hard.
- Sub-cloud crops may lose context if region is wrongly selected.
- Rendering steps double inference cost.
- 3D box / scene-graph format too rigid → model over-fits the template.
- Hard to evaluate intermediate steps (need a proxy for "good grounding").

Mitigations:
- Use existing annotated 3D datasets (ScanNet, SceneVerse) to generate weak CoT.
- Provide a tool that, given a relation triplet, retrieves the supporting sub-cloud (so model only needs to point, not crop).
- Cap the number of multimodal steps per trajectory.

---

## 10. Related Work

- *The Point, the Vision and the Text* (arXiv:2504.04540) — diagnostic.
- PointCoT (arXiv:2602.23945) — text-only CoT for point clouds.
- SceneCOT — grounded 3D-scene CoT.
- 3D-R1 (arXiv:2507.23478) — RL CoT for 3D VLMs.
- Think3D (arXiv:2601.13029) — view-manipulation CoT.
- 3DThinker (arXiv:2510.18632) — latent 3D imagination.
- SpatialReasoner (arXiv:2504.20024) — explicit 3D intermediate.
- SpatialThinker (arXiv:2511.07403) — structured multimodal trace (RGB).
- Visual Sketchpad — visual CoT (2D).
- 3D-LLM, PointLLM-V2, LiDAR-LLM — backbones to build on.

---

## 11. Next Steps

1. Decide whether to start from PointLLM-V2 (object-level) or a scene-level 3D-LLM (ScanNet-scale).
2. Pick benchmark (SQA3D recommended).
3. Build a small annotated CoT subset (V3 format) to bootstrap SFT.
4. Train + evaluate vs baselines.
5. Ablate step types (V1 vs V3 vs hybrid).
