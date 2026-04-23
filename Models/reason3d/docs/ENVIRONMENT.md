# Reason3D environment (GPU cluster)

Target: **Linux**, **NVIDIA GPU**, **CUDA 11.8** aligned with a **cu118** PyTorch wheel (this repo’s `pointgroup_ops` build assumes nvcc major matches `torch.version.cuda`).

## 1. Driver and CUDA runtime

- Install a recent **NVIDIA driver** on the node (enough for your CUDA toolkit / PyTorch).
- Either use **conda** for compile headers (below) or install **CUDA 11.8 toolkit** system-wide if you want `REASON3D_CUDA_HOME=/usr/local/cuda-11.8` with a real `nvcc`.

## 2. Conda environment

Example name: `reason3d310` (Python 3.10).

```bash
conda create -n reason3d310 python=3.10 -y
conda activate reason3d310
```

### PyTorch (CUDA 11.8 wheel)

Install the **cu118** build from the [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) page. Typical pattern:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

Then verify:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
PY
```

**Rule:** `torch.version.cuda` major (e.g. **11.8**) should match the **nvcc** you use to compile `pointgroup_ops`. Mixing PyTorch cu118 with CUDA 12-only nvcc often breaks the extension build.

### Other Python deps

From the Reason3D / LAVIS tree (repo root containing `train.py`):

```bash
cd Models/reason3d
pip install -r requirements.txt   # if present; otherwise install from project README or environment.yml if added
```

Common stack includes: `transformers`, `timm`, `omegaconf`, `einops`, `open3d`, `tensorboard`, `webdataset`, `gorilla`, `pointgroup` stack deps, etc. (match whatever the upstream Reason3D / LAVIS repo documents.)

### CUDA compile headers in conda (often required)

If `nvcc` comes from conda’s `cuda-nvcc` / `cuda-toolkit`, ensure headers are visible (Thrust / CCCL layout):

```bash
conda install -y cuda-nvcc cuda-cudart-dev cuda-cccl=11.8.89 -c nvidia
```

The script `scripts/build_pointgroup_ops.sh` sets `CPATH`/`CUDA_INC_PATH` toward `targets/x86_64-linux/include` when present.

## 3. Build `pointgroup_ops` (required)

```bash
cd Models/reason3d
bash scripts/build_pointgroup_ops.sh
```

If the build fails, read the comments at the top of `scripts/build_pointgroup_ops.sh` (stale `CUDA_HOME`, missing Thrust, CCCL vs nvcc mismatch). Reinstall after env fixes:

```bash
bash scripts/reinstall_pointgroup_ops.sh
```

## 4. Runtime layout

- **`PYTHONPATH`** must include the Reason3D root (`Models/reason3d`), e.g.:

  ```bash
  export PYTHONPATH="/path/to/spatial_reasoning/Models/reason3d:${PYTHONPATH:-}"
  ```

- **Hugging Face caches** (downloaded on first run if not cached): `google/flan-t5-xl`, `bert-base-uncased`, etc. Set `HF_HOME` or `TRANSFORMERS_CACHE` to shared fast storage on the cluster if downloads are slow.

- **Configs** under `lavis/projects/reason3d/` use absolute paths like `/data/scannetpp` and `/data/checkpoints/...` — mirror those paths on the new cluster or override with `--options` (see `docs/DATA_SYNC.md`).

## 5. Quick smoke test

```bash
cd Models/reason3d
python -c "import pointgroup_ops, torch; print('ok', torch.cuda.device_count())"
```

Then a tiny eval (after data sync) using `scripts/run_surprise_zeroshot_eval_small.sh` with `REASON3D_CKPT` set.
