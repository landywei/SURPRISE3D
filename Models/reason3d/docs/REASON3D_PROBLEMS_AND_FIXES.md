# Reason3D on Surprise / ScanNet++: problems we hit and how we fixed them

This document is separate from `REASON3D_FORK_CHANGES.md` (which lists files and git commands). Here we explain **why** things broke and **what** we changed conceptually. The most visible issue for Surprise-style data is the **gap between annotation `object_id` and what appears in the sampled `.pth`** — often summarized as “top instance ids” / instance whitelist behavior.

---

## 1. Annotation `object_id` vs sampled point cloud (the main surprise)

### Symptom

- QA rows in JSON refer to an **`object_id`** (ScanNet++ / Surprise instance id).
- After **area-based subsampling** and **nearest-vertex** transfer of labels onto points, the tensor **`sampled_instance_anno_id`** in `<scene>.pth` often **does not contain** that id.
- Then: **zero GT mask** in the dataset (`gt_pmask` all zeros), **`AssertionError: gt_pmask.int().max() == 1`**, or **nonsense eval** if those rows are still evaluated.

### Why it happens (not a single bug)

Several pipeline stages interact:

1. **Instance whitelist (“top100” style policies)**  
   Many ScanNet++ / Surprise preprocessing paths only keep a **subset** of instance ids (e.g. frequent or “top” instances). Vertices/points tied to instances **outside** that set never receive a usable `objectId` in the label pass that feeds the sampled tensor.

2. **`GetLabelsOnVertices` / semantic–instance rules**  
   Some semantic classes are **ignored** for instance labeling. If your target object’s class is treated as background or ignored, its vertices may **never** get the instance id you expect in the vertex label field that sampling reads.

3. **Sampling itself**  
   **Open3D area sampling** keeps a subset of points; labels follow **nearest mesh vertex**. If the object is rare in the sampled set or id is missing on the mesh side for that vertex, the id can still be absent in **`sampled_instance_anno_id`**.

4. **`sample_factor` alone**  
   Increasing sampling density can help **marginally** but does **not** fix ids that are **excluded by policy** (whitelist / ignore list). Those ids will **never** land in the tensor.

### Fixes we applied in code / config

- **`filter_missing_gt_in_pth: true`** in `dataset_init` (train + eval YAMLs where we care).  
  At dataset init we **`torch.load` each scene’s `.pth` once**, cache the set of instance ids in **`sampled_instance_anno_id`**, and **drop** annotations whose target **`object_id` is not in that set**.  
  - Removes the **assert** on empty GT during training.  
  - Avoids evaluating QA rows with **no supervisable / no checkable** mask.

- **`pth_rel_subdir`** must match where `.pth` files actually live (e.g. **`processed`** under `/nfs-stor/lan.wei/data/scannetpp/processed/`, not a wrong sibling folder). Wrong subdir → glob finds nothing or wrong files → “empty dataset” or inconsistent filtering.

- **Clearer assert message** in `threedrefer_datasets.py` when a row slips through without GT, pointing at **`filter_missing_gt_in_pth`** and **`pth_rel_subdir`**.

### Operational takeaway

Treat **JSON `object_id`** and **`.pth` instance ids** as **two views of the world**. For Surprise + this preprocessing, **always** enable **`filter_missing_gt_in_pth`** unless you are debugging the raw mismatch on purpose. For qualitative work, only rows that survive the filter have a well-defined GT in the sampled cloud.

---

## 2. `pointgroup_ops` build failures (CUDA / Thrust / headers)

### Symptoms

- `ModuleNotFoundError: pointgroup_ops`
- Missing **`cuda_runtime_api.h`**, **Thrust**, or **CCCL** layout mismatches vs **nvcc** version.

### Fixes

- **`scripts/build_pointgroup_ops.sh`**: sanitize bad **`CUDA_HOME`**, add conda **`targets/.../include`** to include paths, align **nvcc major** with **`torch.version.cuda`**, document **`REASON3D_CUDA_HOME`** for a full CUDA **12.x** toolkit when conda headers are not enough.
- **`lib/setup.py`**: optional **`POINTGROUP_CUDA_INCLUDE_DIRS`** for extra include dirs; when **`CONDA_PREFIX`** is set, default system includes prefer the **conda sysroot** `.../x86_64-conda-*/sysroot/usr/include` instead of bare **`/usr/include`**, so conda’s `nvcc` + host compiler are not mixed with a mismatched glibc (avoids e.g. **`bits/timesize.h: No such file or directory`**).
- **`scripts/reinstall_pointgroup_ops.sh`**: clean reinstall loop.

**Rule of thumb:** PyTorch **cu121** wheel → use **CUDA 12–compatible** nvcc and headers; mixing a **CUDA 11** `nvcc` with **cu121** PyTorch (or the reverse) is a common footgun.

---

## 3. Empty dataset after config changes

### Symptom

`AssertionError: Empty dataset` from **`get_sp_filenames()`** (no `*.pth` under the configured root/subdir).

### Cause

**`pth_rel_subdir`** did not match on-disk layout (e.g. points under `.../processed/` but YAML said `scannetpp`).

### Fix

Set **`datasets.3d_refer.dataset_init.pth_rel_subdir: "processed"`** (or your actual folder name) in eval/finetune YAMLs.

---

## 4. SPFormer backbone path and “not a checkpoint” errors

### Symptom

Errors loading **`point_encoder_cfg.pretrained`** — wrong path or file that is not a compatible backbone checkpoint.

### Fix

Point **`point_encoder_cfg.pretrained`** at the real **SPFormer / UNET** weight file (in our runs **`/nfs-stor/lan.wei/data/checkpoints/spf_scannet_512.pth`**), distinct from the **full Reason3D** `.pth`.

---

## 5. Python / Hub / hardcoded paths in the model

### Symptoms

- Crash on **`import ipdb`**
- Invalid **`/workspace/huggingface/...`** paths for BERT / T5

### Fixes

- Removed unused **`ipdb`** import from **`reason3d_t5.py`**.
- **BERT** from Hub id **`bert-base-uncased`**; **T5** from **`google/flan-t5-xl`** (or cache via **`HF_HOME`**), no remapping to non-existent `/workspace/...`.

---

## 6. Eval checkpoint path: “checkpoint url or path is invalid”

### Symptom

**`RuntimeError: checkpoint url or path is invalid`** when loading full Reason3D weights.

### Causes

- **`REASON3D_CKPT`** not a real file (typo, relative path from wrong cwd, stray quotes).
- Overwriting **`model.pretrained`** with the Reason3D `.pth` while the merged default was still the **BLIP2** URL — confusing three different “pretrained” concepts (BLIP2 default, SPFormer backbone, full Reason3D).

### Fixes

- **`model.reason3d_checkpoint`** for the **full Reason3D** `.pth`; scripts pass that key.
- **`load_from_pretrained`**: **`expanduser`**, strip quotes, **`abspath`** for relative paths, clearer error with **resolved path + cwd**.
- **`base_model.py`**: fix broken **`assert`** when `pretrained` is missing.

---

## 7. Small eval scene allowlist: “kept 0 / N annotations”

### Symptom

**`eval scene allowlist kept 0 / …`** then runtime error: no overlap between allowlist and val JSON.

### Cause

**`trial_scenes.txt`** contained **ScanNet++ scene hashes** that do **not** appear in **`surprise_val.json`** (different split / id set).

### Fix

Replaced trial ids with **`scene_id` values actually present** in the Surprise val JSON; added an **early error** if allowlist ∩ annotation scenes is empty, with **example ids** from both sides.

---

## 8. “Building datasets” for many minutes

### Symptom

Log sits on **“Building datasets…”** for 5–15+ minutes.

### Cause

With **`filter_missing_gt_in_pth`**, init does **one full `torch.load` per unique scene** (first time that scene is seen) to read **`sampled_instance_anno_id`** and decide which QA rows to keep. Hundreds of large **`.pth`** files on **network / HDD** storage is slow, but **runs once per process** unless you restart workers.

### Fixes

1. **Observability:** a log line states **how many unique scenes** may be read so a long pause is not mistaken for a hang.

2. **Persisted instance-id cache (recommended on clusters):** set in **`dataset_init`**:

   - **`instance_id_cache_file`**: path to a JSON file (e.g. under `processed/`, same disk as `.pth`).
   - **`instance_id_cache_write: true`** on **one** run that is allowed to read all `.pth` files; the file stores `scene_id → list of instance ids` plus **`missing_pth`** for scenes with no file.
   - Set **`instance_id_cache_write: false`** on later runs; the dataset **loads the JSON** and filters **without** re-`torch.load` every scene (only loads `.pth` for scenes **missing** from the cache, e.g. new scenes).

   Invalidate or delete the cache file if you **regenerate** `.pth` files or change **`pth_rel_subdir`** / **`pts_root`** (the file records those paths and refuses to load if they mismatch).

3. **Optional filtered JSON dump:** **`write_filtered_annotations_to`** writes the **post-filter** annotation list to a JSON file (rank 0 only). You can point **`build_info.annotations.train.storage`** at that file for reproducibility; **still** use the instance cache for fast init unless you also disable **`filter_missing_gt_in_pth`** (not recommended if `.pth` can change).

---

## 9. Qualitative eval outputs

### Need

Save **per-sample predictions** (masks + metadata) for offline analysis.

### Fix

- **`run.save_eval_predictions`** + implementation in **`refer_seg_task.py`**: under each run’s **`lavis/output/.../qualitative/`**, write **`predictions.jsonl`** and **`masks/*.npz`**.  
- **`lavis/output/`** is **gitignored**; sync with **`rsync`** (see **`DATA_SYNC.md`**).

---

## 10. Summary table

| Problem area | Typical symptom | Mitigation |
|--------------|-----------------|------------|
| Instance id / top policy vs `.pth` | No GT in sampled cloud, bad eval / assert | **`filter_missing_gt_in_pth`**, correct **`pth_rel_subdir`**, understand whitelist / ignores |
| CUDA extension build | Missing headers / wrong nvcc vs torch | **`build_pointgroup_ops.sh`**, conda CUDA dev packages, **`REASON3D_CUDA_HOME`** |
| Wrong `.pth` directory | Empty dataset | **`pth_rel_subdir`** |
| Backbone vs full model | Wrong load / path errors | **`point_encoder_cfg.pretrained`** vs **`model.reason3d_checkpoint`** |
| Hub / ipdb / `/workspace` | Import or path errors | Hub ids, remove **`ipdb`**, no hardcoded cache roots |
| Allowlist scenes | 0 rows kept | Allowlist **`scene_id`** from **the same JSON** as eval |
| Long dataset init | Stuck “building” | Expected with filter + many scenes; see log line |
| Qualitative artifacts | Nothing on disk | **`save_eval_predictions`**, sync **`lavis/output/`** |

If you extend preprocessing (e.g. change top-instance policy), revisit **`filter_missing_gt_in_pth`** and document your new id semantics in the same place you version the JSON.
