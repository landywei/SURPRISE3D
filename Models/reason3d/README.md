<p align="center">
  <h1 align="center">Reason3D: Searching and Reasoning 3D Segmentation via Large Language Model [3DV 2025]
  </h1>
  <p align="center">
    <a href="https://kuanchihhuang.github.io/"><strong>Kuan-Chih Huang</strong></a>,
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>,
    <a href="https://luqi.info/"><strong>Lu Qi</strong></a>,
    <a href="https://yanshuicheng.info/"><strong>Shuicheng Yan</strong></a>,
    <a href="https://faculty.ucmerced.edu/mhyang/"><strong>Ming-Hsuan Yang</strong></a>
  </p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.17427-red)](https://arxiv.org/abs/2405.17427)
[![Project](https://img.shields.io/badge/project-page-green)](https://kuanchihhuang.github.io/project/reason3d/)

</div>

## Installation

1. Create conda environment. We use `python=3.8` `pytorch=1.11.0` and `cuda=11.3`.
```bash
conda create -n reason3d python=3.8
conda activate reason3d
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

2. Install [LAVIS](https://github.com/salesforce/LAVIS)
```bash
git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS
cd SalesForce-LAVIS
pip install -e .
```

3. Install segmentor from this [repo](https://github.com/Karbo123/segmentator) (used for superpoint construction). We also provide an alternative PyTorch implementation `segmentator_pytorch.py`, though it may yield slightly lower performance.


4. Install pointgroup_ops
```bash
cd lavis/models/reason3d_models/lib
sudo apt-get install libsparsehash-dev
python setup.py develop
```

## Data Preparation

### ScanNet++ Dataset

Download the [ScanNet++](https://github.com/scannetpp/scannetpp) dataset.

Put the downloaded scenes as follows.
```
datasets
├── data
│   ├── 0a5c013435
│   ├── 0a7cc12c0e
│   ├── ...
├── metadata
├── splits
```

Follow [ScanNet++](https://github.com/scannetpp/scannetpp) to Prepare 3D Semantics Training Data, you have to modify the semantic/configs/prepare_training_data.yml. To ensure all categories are included, replace the original instance_class.txt file with the one we provide:
```
cd /your_path_to_scannetpp/scannetpp
python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
```

After running the script, the scannet++ dataset structure should look like below.
```
datasets
├── data
│   ├── 0a5c013435
│   ├── 0a7cc12c0e
│   ├── ...
├── metadata
├── splits
├── processed
│   ├── 0a5c013435.pth
│   ├── 0a7cc12c0e.pth
│   ├── ...
```
Follow [UniDet3D](https://github.com/filaPro/unidet3d/) to generate superpoints. You can also use other methods to generate superpoint on your own.
```
cd /your_path_to_UniDet3D/UniDet3D/data/scannetpp
python preprocess_raw_data.py --path_to_data path_to_dataset --output_dir path_to_save_preprocessed_raw_data
```

Add superpoints to pth file.
```
python update_superpoints.py --pth_dir datasets/data/processed --scene_dir path_to_save_preprocessed_raw_data
```
### Surprise3D dataset

Download [Surprise3D](https://huggingface.co/datasets/hhllzz/surprise-3d) annotations.

```
datasets
├── surprise-3d-train.json
├── surprise-3d-val.json
```

## Pretrained Backbone
Download the [SPFormer](https://github.com/sunjiahao1999/SPFormer) pretrained backbone (or provided by [3D-STMN](https://github.com/sosppxo/3D-STMN)) and move it to checkpoints.
```
mkdir checkpoints
mv ${Download_PATH}/sp_unet_backbone.pth checkpoints/
```
You can also pretrain the backbone by yourself and modify the path [here](lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml#L15).

## Training

- **Surprise3D finetune (recommended):** From the repo root of this baseline, build ops once (`bash scripts/build_pointgroup_ops.sh`), then run the wrapper (see `docs/finetune_eval_scripts.md` for env vars such as `REASON3D_INIT_CKPT`, `REASON3D_PTS_ROOT`, `REASON3D_PTH_SUBDIR`):

```bash
REASON3D_INIT_CKPT=<path_to_reason3d_pretrained.pth> bash scripts/run_surprise_finetune.sh
```

Configs for **geo** (geometry-aware) and **chain** variants: `scripts/run_surprise_finetune_geo.sh`, `scripts/run_surprise_finetune_chain.sh`, etc.

- **ScanRefer-style scratch config (legacy YAML in this fork):**

```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 train.py --cfg-path lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml
```

## Evaluation

- **Surprise3D val (recommended):**

```bash
REASON3D_CKPT=<path_to_checkpoint.pth> bash scripts/run_surprise_zeroshot_eval.sh
```

Small / geo eval wrappers: `scripts/run_surprise_zeroshot_eval_small.sh`, `scripts/run_surprise_zeroshot_eval_small_geo.sh`, `scripts/run_surprise_zeroshot_eval_small_chain.sh`.

- **Legacy val YAML:**

```bash
python evaluate.py --cfg-path lavis/projects/reason3d/val/reason3d_scanrefer_scratch.yaml --options model.pretrained=<path_to_pretrained_checkpoint> run.save_results=True
```

Note: inference is typically run with batch size 1. Add `run.save_results=True` (or use the eval scripts’ save flags) if you want prediction dumps on disk.


## Visualization

You can visualize prediction results using:
```
python visualize.py --idx <sample_index> --result_dir <results_directory>
```
`<sample_index>`: Index of the sample you wish to display. `<results_directory>`: Path to either the `reason_preds` or `refer_preds` directory containing the results.


## Citation

If you find our work useful for your project, please consider citing our paper:


```bibtex
@article{reason3d,
  title={Reason3D: Searching and Reasoning 3D Segmentation via Large Language Model},
  author={Kuan-Chih Huang and Xiangtai Li and Lu Qi and Shuicheng Yan and Ming-Hsuan Yang},
  journal={3DV},
  year={2025}
}
```
