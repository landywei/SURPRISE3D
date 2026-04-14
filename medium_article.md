# Surprise3D: Can AI Really Understand "Behind the Chair"? Our NeurIPS 2025 Benchmark Says Not Yet.

*We built a 200k-query dataset to test whether 3D AI models truly understand spatial relationships — the results were humbling*

---

Ask any 3D vision model to "find the red chair next to the table" and it performs reasonably well. But ask it to "find the object behind the chair" — without naming what that object is — and even the best models collapse to near-random accuracy.

This gap between **semantic recognition** and **genuine spatial reasoning** is what motivated us to build **Surprise3D**, a large-scale benchmark we presented at NeurIPS 2025 (Datasets & Benchmarks Track).

> **Paper**: [arXiv:2507.07781](https://arxiv.org/abs/2507.07781)
> **Dataset**: [HuggingFace](https://huggingface.co/datasets/hhllzz/surprise-3d)
> **Code**: [github.com/liziwennba/SURPRISE3D](https://github.com/liziwennba/SURPRISE3D)

---

## The Shortcut Problem in 3D Vision-Language Research

Here's an uncomfortable truth about existing 3D vision-language benchmarks: most of them accidentally reward **name-matching** rather than **spatial understanding**.

Consider a typical query in datasets like ScanRefer: "The brown wooden chair on the left side of the desk." A model can often get this right by simply detecting all "chairs" and picking the "brown" one near a "desk." The spatial cue ("on the left side") is barely needed — the semantic cues do the heavy lifting.

We verified this experimentally. When we stripped object names from existing benchmark queries, strong models like ScanRefer and EDA saw their accuracy **plummet from ~40% to under 10%**. These models weren't reasoning about space — they were matching keywords.

---

## What Makes Surprise3D Different

We designed Surprise3D with one principle: **force the model to actually reason about spatial relationships**.

### The Dataset

Built on top of **ScanNet++ v2** (900+ high-quality indoor scenes), Surprise3D contains **200k+ vision-language pairs** with **89k+ human-annotated spatial queries**. The critical design choice: queries **never name the target object**.

Instead of "find the red chair," you get queries like:

- **Relative Position**: "Find the object behind the chair"
- **Narrative Perspective**: "Entering the door, with your back to the door, the object on your left side"
- **Parametric Perspective**: "At the perspective (3.00, -0.76, -1.24, ...), what's the closest object to sit?"
- **Absolute Distance**: "A chair that is 2.63 meters away from the balcony door"

Each category tests a different aspect of spatial intelligence. Relative position is about understanding how objects relate to each other. Narrative perspective requires understanding a described viewpoint. Parametric perspective uses explicit coordinates. Absolute distance demands metric reasoning.

### Two Annotation Pipelines

Building 200k+ high-quality spatial queries required two complementary approaches:

**Pipeline 1 — Human Spatial Annotation (89k+ queries):** Volunteers go through a structured 3-step process. First, they select a 3D scene. Then they write spatial descriptions under two scenarios: in *Situation 1*, they describe their own orientation in the scene and an object's position relative to that viewpoint (this produces narrative and parametric perspective queries); in *Situation 2*, they describe spatial relationships between objects (relative position and absolute distance queries). Finally, they verify the correct target object in the 3D scene. This human-in-the-loop process ensures the spatial queries are grounded in real 3D geometry and are genuinely unambiguous.

**Pipeline 2 — Automatic Generation with Quality Control (knowledge queries):** For common-sense and human-intention queries, we use an LLM to generate initial Q&A pairs from scene metadata and object frequencies. These go through an automated quality-control stage where a second model identifies problematic queries — for instance, cases where the question asks about hanging a towel but the answer lists the towel itself rather than the rack. Flagged queries are rewritten and then verified by human annotators. We also specifically augment query generation for rare objects to address the long-tail distribution problem.

---

## What the Benchmark Reveals

We evaluated a range of methods on our 3D-SRS (3D Spatial Reasoning Segmentation) protocol, including 3D-Vista, ChatScene, Intent3D, Reason3D, and our own MLLM-For3D. The findings paint a clear picture:

**Zero-shot performance is near-random.** When models haven't been trained on spatial reasoning queries, they essentially guess. Average Acc@0.25 across methods hovers around 8%, Acc@0.50 around 5%.

**Fine-tuning helps, but not enough.** Training on Surprise3D queries improves results, but models still struggle significantly. The biggest gains appear in "easier" categories like relative position, while narrative perspective and parametric reasoning remain extremely challenging.

**Knowledge reasoning is easier than spatial reasoning.** Queries involving common sense ("Where do people most often sit when watching TV?") and human intention ("I want to watch the news, what should I turn on?") fare better than pure spatial queries — likely because they can leverage the language model's world knowledge.

These results confirm that **spatial reasoning remains a fundamental bottleneck** in 3D scene understanding, and current architectures don't have a good inductive bias for it.

---

## Why This Matters

Spatial reasoning is foundational for:

- **Robotics**: "Put the cup on the shelf behind you" requires understanding egocentric spatial relationships
- **AR/VR navigation**: "The exit is 3 meters to your left" demands metric spatial awareness
- **Assistive technology**: Helping visually impaired users understand room layouts requires robust spatial language grounding

Without benchmarks that specifically isolate spatial reasoning from semantic shortcuts, we can't measure — or improve — this capability. Surprise3D is our contribution toward closing that gap.

---

## Using Surprise3D in Your Research

### Quick Start

```python
from datasets import load_dataset
dataset = load_dataset("hhllzz/surprise-3d")
```

The dataset includes train (180k rows) and validation (10.2k rows) splits. Each entry contains the scene ID, object ID, spatial query, question type, and reference information.

You'll also need [ScanNet++ v2](https://kaldir.vc.in.tum.de/scannetpp/) for the 3D scene data (requires separate access request).

### Baselines

We provide modified implementations of [Reason3D](https://github.com/KuanchihHuang/Reason3D) and [Intent3D](https://github.com/WeitaiKang/Intent3D) that work with Surprise3D out of the box. See the [Models/](https://github.com/liziwennba/SURPRISE3D/tree/main/Models) directory for details.

---

## Companion Work: MLLM-For3D

Surprise3D is one of two related papers we published at NeurIPS 2025. The other is **MLLM-For3D**, which tackles the inverse problem: how to build better 3D reasoning segmentation models by transferring capabilities from 2D vision-language models to 3D, without any 3D annotations.

> [MLLM-For3D Paper](https://arxiv.org/abs/2503.18135) | [Code](https://github.com/tmllab/2025_NeurIPS_MLLM-For3D) | [Project Page](https://tmllab.github.io/2025_NeurIPS_MLLM-For3D/)

Together, these two works address the problem from both sides: Surprise3D gives us a rigorous way to **measure** spatial reasoning, and MLLM-For3D provides a strong approach to **improve** it.

---

## Citation

```
@inproceedings{huang2025surprise3d,
  title={SURPRISE3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes},
  author={Huang, Jiaxin and Li, Ziwen and Zhang, Hanlue and Chen, Runnan and Gao, Zhengqing and He, Xiao and Guo, Yandong and Wang, Wenping and Liu, Tongliang and Gong, Mingming},
  booktitle={NeurIPS, Datasets and Benchmarks Track},
  year={2025}
}
```

---

*Jiaxin Huang is a researcher at MBZUAI working on 3D scene understanding and multimodal AI. Follow for more updates on 3D vision-language research.*

**Tags**: `#MachineLearning` `#ComputerVision` `#NeurIPS` `#3DSceneUnderstanding` `#Dataset` `#Benchmark` `#SpatialReasoning` `#DeepLearning`
