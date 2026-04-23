# Reason3D environment (GPU cluster)

Target: **Linux**, **NVIDIA GPU**, **CUDA 12.x** aligned with a **cu121** PyTorch wheel (this repo’s `pointgroup_ops` build assumes **`nvcc` major** matches **`torch.version.cuda`**, e.g. both **12**).

## 1. Driver and CUDA runtime

- Install a recent **NVIDIA GPU driver** on the node (sufficient for CUDA 12.x runtimes; newer drivers are fine).
- For **compiling** `pointgroup_ops`, use **`nvcc` 12.x** (e.g. conda’s `cuda-nvcc` 12.1+, or `REASON3D_CUDA_HOME=/usr/local/cuda-12.x`). If the default `nvcc` on `PATH` is **not** the toolkit you want, prepend that toolkit’s `bin` or set **`REASON3D_CUDA_HOME`** so the build does not pick a mismatched major.

## 2. Conda environment

Example name: `reason3d310` (Python 3.10).

```bash
conda create -n reason3d310 python=3.10 -y
conda activate reason3d310
```

### PyTorch (CUDA 12.1 / cu121)

`requirements.txt` pins **`setuptools<82`** because **setuptools 82+** drops **`pkg_resources`**, which **PyTorch 2.1**’s `torch.utils.cpp_extension` still imports when building CUDA extensions.

Install **CUDA 12.1.1 + conda sparsehash + Python deps** in one step (conda env active, **login** node with network):

```bash
cd Models/reason3d
bash scripts/install_reason3d_deps.sh
```

That script tries **`conda-forge::sparsehash`** after the CUDA toolkit. If conda fails (common with broken `nvidia::*` metadata), set **`REASON3D_SKIP_SPARSEHASH=1`** and run **`bash scripts/prep_google_sparsehash_headers.sh`** once (see below).

Or install the **cu121** PyTorch builds only from the [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) page. With **internet** (use **`--extra-index-url`** so `pip` still resolves packages from PyPI):

```bash
cd Models/reason3d
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

Or install the torch stack only:

```bash
pip install "numpy<2" torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

To repair a broken mixed install (recommended one-liner on shared envs):

```bash
bash scripts/reinstall_pytorch_stack_cu121.sh
```

Then verify:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
PY
```

**Rule:** `torch.version.cuda` (e.g. **12.1**) should match the **nvcc** major you use to compile `pointgroup_ops`. Do not compile the extension with a different CUDA major (e.g. **11** or **13**) while using PyTorch **cu121**.

### Other Python deps

`requirements.txt` lists fork deps plus LAVIS imports used here (`webdataset`, `omegaconf`, `iopath`, `decord`, `timm`, `transformers==4.33.2`, **`spconv-cu121`**, …). **`spconv-cu121`** supplies **`import spconv.pytorch`** for CUDA 12.1 (matches **cu121** PyTorch). **`torch-scatter`** is required by this codebase but installed from **PyG wheels** in `scripts/install_reason3d_deps.sh` (and refreshed in `scripts/reinstall_pytorch_stack_cu121.sh`) using **`https://data.pyg.org/whl/torch-2.1.2+cu121.html`** — not PyPI.

**You do *not* need** `torch-sparse` or `torch-cluster` for this repo (nothing imports them). Prefer **`transformers==4.33.2`** as pinned; newer versions (e.g. 4.36.x) are untested with this fork.

**PyTorch URL:** use **`https://download.pytorch.org/whl/cu121`**, not `https://pytorch.org` (that host is not the pip wheel index).

### CUDA compile toolkit in conda (nvcc 12.1, matches PyTorch cu121)

Prefer a **single pinned NVIDIA label** so `nvcc`, `cuda-cudart-dev`, and CCCL headers all come from **CUDA 12.1.1** (aligned with **torch 2.1.2+cu121**, `torch.version.cuda` **12.1**):

```bash
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
```

This is what **`scripts/install_reason3d_deps.sh`** runs before `pip install -r requirements.txt` (unless `REASON3D_SKIP_CONDA_CUDA=1`). For the previous **12.1.0** bundle from the same series, use `-c nvidia/label/cuda-12.1.0`.

The script `scripts/build_pointgroup_ops.sh` sets `CPATH`/`CUDA_INC_PATH` toward `targets/x86_64-linux/include` when present.

### Google sparsehash headers (`#include <google/dense_hash_map>`)

HPC **compute** images may omit `libsparsehash` under `/usr`. Install headers into the same conda env (e.g. on a **login** node with network):

```bash
conda install -c conda-forge sparsehash
```

**If `conda` errors with** `InvalidSpec: ... nvidia/...::cuda-compiler==...` **is not available** (including with `--no-deps`): the environment still records a CUDA metapackage build that is **not on the current channel index**. Conda refuses *any* install for that prefix until the bad record is fixed. Options:

1. **Unblock conda** (only if you understand what CUDA packages this env should keep): list and remove the broken metapackage, then reinstall what you need:  
   `conda list -p "$CONDA_PREFIX" | grep -E 'cuda|nvidia'`  
   `conda remove -p "$CONDA_PREFIX" --force cuda-compiler`  (or other `nvidia::*` packages that are inconsistent — coordinate with the env owner for shared `reason3d310`.)
2. **Do not use conda for sparsehash** (recommended on broken envs): download **upstream** headers (once, needs network on **login**):  
   `cd Models/reason3d && bash scripts/prep_google_sparsehash_headers.sh`  
   That downloads the release, extracts under `Models/reason3d/third_party/sparsehash-upstream-2.0.4/`, then runs **`./configure` and `make`** to generate `src/sparsehash/internal/sparseconfig.h` (required by `#include <sparsehash/...>`; the raw tarball does not include it). **Login node:** needs a normal build toolchain (`gawk`, C/C++ compiler). The build script `build_pointgroup_ops.sh` then sets `POINTGROUP_SPARSEHASH_ROOT` to the `…/src` directory automatically if the tree exists, or you can:  
   `export POINTGROUP_SPARSEHASH_ROOT="$PWD/third_party/sparsehash-upstream-2.0.4/src"`.  
   **No-internet GPU nodes:** run `prep_google_sparsehash_headers.sh` **once** on a **login** (or any host with network) so `third_party/` lives on your **shared** path (e.g. NFS). On the compute node, `bash scripts/build_pointgroup_ops.sh` does not need the network for the sparsehash download; it compiles from local sources. Ensure the conda env and Python deps are already installed (`pip` does not need to reach PyPI for a typical local `pip install -e .` on this package when build isolation is off).
3. Shorter path if conda still works: **`conda install -c conda-forge sparsehash`**, or **`--no-deps`** only if (1) does not apply and the env is consistent.

4. **Mamba** can sometimes solve where classic conda fails: `mamba install -c conda-forge sparsehash` (if installed); it can hit the same `InvalidSpec` if the **installed** `cuda-compiler` record is bad.

`pointgroup_ops` `setup.py` searches `CONDA_PREFIX` and the conda sysroot for a directory that contains `google/dense_hash_map`. If the headers are elsewhere, set the **parent of `google/`** (one directory level above the `google` folder):

```bash
export POINTGROUP_SPARSEHASH_ROOT=/path/to/parent_of_google
```

Optional extra include paths: `POINTGROUP_CUDA_INCLUDE_DIRS` (see `lavis/models/reason3d_models/lib/setup.py`).

**Disk quota:** if `~/.cache/pip` fills your home volume, set e.g. `export PIP_CACHE_DIR=/path/to/large/disk/pip_cache` before `pip install`.

## 3. Build `pointgroup_ops` (required)

**If `nvcc -V` does not match `torch.version.cuda`:** your shell is picking the wrong toolkit (old module path, stale `CUDA_HOME`, or conda not first on `PATH`).

```bash
conda activate reason3d310
type -a nvcc
python - <<'PY'
import torch, re
v = torch.version.cuda or ""
m = re.match(r"(\d+)", v)
print("torch.version.cuda major:", m.group(1) if m else "?")
PY
nvcc -V
# Put the matching nvcc first, e.g. conda or /usr/local/cuda-12.x:
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
# Or: export REASON3D_CUDA_HOME=/usr/local/cuda-12.6
```

```bash
cd Models/reason3d
bash scripts/build_pointgroup_ops.sh
```

If the build fails, read the comments at the top of `scripts/build_pointgroup_ops.sh` (stale `CUDA_HOME`, wrong `nvcc` vs `torch.version.cuda`, missing Thrust). Reinstall after env fixes:

```bash
bash scripts/reinstall_pointgroup_ops.sh
```

## 4. Runtime layout

- **`PYTHONPATH`** must include the Reason3D root (`Models/reason3d`), e.g.:

  ```bash
  export PYTHONPATH="/path/to/spatial_reasoning/Models/reason3d:${PYTHONPATH:-}"
  ```

- **Hugging Face caches** (downloaded on first run if not cached): `google/flan-t5-xl`, `bert-base-uncased`, etc. Set `HF_HOME` or `TRANSFORMERS_CACHE` to shared fast storage on the cluster if downloads are slow.

- **Configs** under `lavis/projects/reason3d/` use absolute paths like `/nfs-stor/lan.wei/data/scannetpp` and `/nfs-stor/lan.wei/data/checkpoints/...` — mirror those paths on the new cluster or override with `--options` (see `docs/DATA_SYNC.md`).

## 5. Quick smoke test

After `pip install` (and after `pointgroup_ops` is built, if you want that checked too):

```bash
cd Models/reason3d
bash scripts/smoke_env.sh
```

Minimal GPU check:

```bash
cd Models/reason3d
python -c "import pointgroup_ops, torch; print('ok', torch.cuda.device_count())"
```

Then a tiny eval (after data sync) using `scripts/run_surprise_zeroshot_eval_small.sh` with `REASON3D_CKPT` set.
