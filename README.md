<div align="center">

# Surprise3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes

### NeurIPS 2025 (Datasets & Benchmarks Track)

**Jiaxin Huang**<sup>1,2</sup>, **Ziwen Li**<sup>3</sup>, **Hanlue Zhang**<sup>3</sup>, **Runnan Chen**<sup>1</sup>, **Zhengqing Gao**<sup>4</sup>, **Xiao He**<sup>2</sup>, **Yandong Guo**<sup>2</sup>, **Wenping Wang**<sup>4</sup>, **Tongliang Liu**<sup>3&dagger;</sup>, **Mingming Gong**<sup>5&dagger;</sup>

<sup>1</sup>MBZUAI &nbsp; <sup>2</sup>AI2Robotics &nbsp; <sup>3</sup>The University of Sydney &nbsp; <sup>4</sup>Texas A&M University &nbsp; <sup>5</sup>The University of Melbourne

<sup>&dagger;</sup>Corresponding authors

[![Paper](https://img.shields.io/badge/arXiv-2507.07781-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.07781)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Dataset-blue)](https://huggingface.co/datasets/hhllzz/surprise-3d)
[![Project Page](https://img.shields.io/badge/Project-Page-4c8dae)](https://liziwennba.github.io/SURPRISE3D)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-7c3aed)](https://neurips.cc)
[![MLLM-For3D](https://img.shields.io/badge/Companion-MLLM--For3D-green)](https://github.com/tmllab/2025_NeurIPS_MLLM-For3D)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/liziwennba/SURPRISE3D?style=social)](https://github.com/liziwennba/SURPRISE3D)

<p>
<a href="assets/MBZUAI.png"><img src="assets/MBZUAI.png" height="50" alt="MBZUAI"></a>&nbsp;&nbsp;
<a href="assets/ai2robotics.png"><img src="assets/ai2robotics.png" height="50" alt="AI2Robotics"></a>&nbsp;&nbsp;
<a href="assets/sydney.jpg"><img src="assets/sydney.jpg" height="50" alt="USyd"></a>&nbsp;&nbsp;
<a href="assets/texasa&muniversity.jpg"><img src="assets/texasa&muniversity.jpg" height="50" alt="Texas A&M"></a>&nbsp;&nbsp;
<a href="assets/Melbourne.png"><img src="assets/Melbourne.png" height="50" alt="UMelb"></a>
</p>

</div>

---

## TL;DR

**Surprise3D** is a large-scale benchmark for evaluating **language-guided spatial reasoning segmentation** in 3D scenes. Unlike existing datasets, our queries **exclude object names** to force genuine spatial reasoning — no shortcut biases.

| | Surprise3D |
|:--|:--|
| 🏠 Scenes | 900+ indoor scenes (ScanNet++ v2) |
| 💬 Queries | 200k+ vision-language pairs |
| ✍️ Human annotations | 89k+ spatial queries (no object names) |
| 🏷️ Object classes | 2,800+ unique classes |
| 🧠 Reasoning types | Relative position · Narrative perspective · Parametric perspective · Absolute distance |

> 🔗 **Companion method**: [**MLLM-For3D**](https://github.com/tmllab/2025_NeurIPS_MLLM-For3D) — adapts 2D MLLMs for label-free 3D reasoning segmentation (NeurIPS 2025). [[Paper]](https://arxiv.org/abs/2503.18135)

---

## Why Surprise3D?

Current 3D vision-language datasets let models take **semantic shortcuts** — they match object names in queries rather than truly reasoning about spatial relationships. We found that even strong multimodal models **drop to near-zero accuracy** on pure spatial reasoning tasks.

**Surprise3D fixes this** by:
1. Crafting queries that describe spatial relationships **without naming the target object**
2. Covering four distinct spatial reasoning skills with human-written queries
3. Providing a standardized 3D Spatial Reasoning Segmentation (3D-SRS) benchmark

### Spatial Reasoning Types

| Type | Example Query |
|:--|:--|
| **Relative Position** | *"Find the object behind the chair"* |
| **Narrative Perspective** | *"Locate the object visible from the sofa"* |
| **Parametric Perspective** | *"Select the object 2 meters to the left of the table"* |
| **Absolute Distance** | *"Identify the object exactly 3 meters in front of you"* |

---

## Annotation Pipeline

Surprise3D is built through **two complementary annotation pipelines**:

### 1. Spatial Annotation (Human)

![Spatial Annotation Pipeline](assets/spatial_annotation_pipline%20(1).pdf)

Volunteers follow a 3-step process: (1) select a 3D scene, (2) write spatial descriptions under two situations — *Situation 1*: describe orientation and object position relative to a viewpoint (narrative/parametric perspective); *Situation 2*: describe spatial relationships between objects (relative position, absolute distance), and (3) verify the correct target objects in the 3D scene.

### 2. Automatic Pipeline (LLM + Human Verification)

![Automatic Pipeline](assets/automatic_pipeline%20(1).png)

For knowledge-based queries (common sense, human intention): an LLM generates initial Q&A pairs from scene metadata, then automated quality control with human-in-the-loop verification identifies and corrects problematic queries.

---

## Data Analysis

![Data Analysis](assets/data_analysis.png)

Balanced coverage across reasoning types with augmentation for low-frequency objects to reduce bias.

---

## Quick Start

### Step 1: Download Annotations

```python
# Option A: Via HuggingFace datasets library
from datasets import load_dataset
dataset = load_dataset("hhllzz/surprise-3d")
```

Or download directly: [🤗 HuggingFace — hhllzz/surprise-3d](https://huggingface.co/datasets/hhllzz/surprise-3d)

**Dataset splits**: Train (180k rows) · Validation (10.2k rows)

**Fields**: `object_id`, `object_name`, `description`, `scene_id`, `question_type`, `reference_id`

### Step 2: Download ScanNet++ Point Clouds

Surprise3D annotations are built on [ScanNet++ v2](https://kaldir.vc.in.tum.de/scannetpp/). You need to request access and download the 3D scene data separately.

### Step 3: Preprocess for Training

```bash
# Clone this repo
git clone https://github.com/liziwennba/SURPRISE3D.git
cd SURPRISE3D

# Preprocess ScanNet++ data for Surprise3D
# See Models/reason3d/ for detailed preprocessing scripts
python Models/reason3d/preprocess_scannetpp.py --data_root /path/to/scannetpp
```

### Step 4: Train & Evaluate Baselines

We provide modified baseline implementations adapted for Surprise3D:

**Reason3D** (3D reasoning segmentation):
```bash
cd Models/reason3d
# See Models/reason3d/README.md for full setup, training, and evaluation
python -m torch.distributed.run --nproc_per_node=4 train.py \
  --cfg-path lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml
```

**Intent3D** (intention-based 3D grounding):
```bash
cd Models/intent3d
# See Models/intent3d/README.md for full setup
conda env create -f environment.yml --name Intent3D
sh scripts/train_scanintend.sh
```

See individual READMEs under [Models/](Models/) for detailed instructions.

---

## Benchmark Results (3D-SRS)

Baseline methods evaluated on Surprise3D under the 3D Spatial Reasoning Segmentation protocol:

### Zero-shot Evaluation

| Method | mIoU | Acc@0.25 | Acc@0.50 |
|:--|:--:|:--:|:--:|
| 3D-Vista | — | — | — |
| ChatScene | — | — | — |
| MLLM-For3D | — | — | — |

### Fine-tuned on Surprise3D

| Method | mIoU | Acc@0.25 | Acc@0.50 |
|:--|:--:|:--:|:--:|
| Intent3D | — | — | — |
| Reason3D | — | — | — |
| **MLLM-For3D** | **—** | **—** | **—** |

<!-- TODO: Fill in exact numbers from arXiv:2507.07781 Table 1 & 2 -->

---

## Code Structure

```
SURPRISE3D/
├── Models/
│   ├── reason3d/            # Modified Reason3D for Surprise3D training/eval
│   │   ├── train.py         # Distributed training script
│   │   ├── evaluate.py      # Evaluation script
│   │   ├── visualize.py     # Visualization of predictions
│   │   └── update_superpoints.py
│   └── intent3d/            # Modified Intent3D baseline
│       ├── train_dist_mod.py
│       ├── prepare_data.py
│       └── environment.yml
├── assets/                  # Figures, logos, data analysis plots
│   ├── task.png             # Task overview figure
│   ├── data_analysis.png    # Dataset statistics
│   └── automatic_pipeline (1).png  # Annotation pipeline
├── LICENSE                  # MIT License
└── README.md
```

---

## Citation

If you use Surprise3D in your research, please cite:

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

@inproceedings{huang2025mllmfor3d,
  title     = {MLLM-For3D: Adapting Multimodal Large Language Model
               for 3D Reasoning Segmentation},
  author    = {Huang, Jiaxin and Chen, Runnan and Li, Ziwen and
               Gao, Zhengqing and He, Xiao and Guo, Yandong and
               Liu, Tongliang and Gong, Mingming},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

## Acknowledgements

We thank the authors of [Reason3D](https://github.com/KuanchihHuang/Reason3D) for their outstanding work. We also thank the [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) team for the real 3D indoor scene data.

## Contact

For questions or collaborations:

- **Jiaxin Huang** — [jiaxin.huang@mbzuai.ac.ae](mailto:jiaxin.huang@mbzuai.ac.ae)
- **Ziwen Li** — [ziwen.li@mbzuai.ac.ae](mailto:ziwen.li@mbzuai.ac.ae)
- **Hanlue Zhang** — [hanlue.zhang@mbzuai.ac.ae](mailto:hanlue.zhang@mbzuai.ac.ae)
