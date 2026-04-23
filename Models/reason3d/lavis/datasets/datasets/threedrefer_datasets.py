"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import json
import torch
import numpy as np
import os.path as osp
import pointgroup_ops
import math
from PIL import Image
from PIL import ImageFile
import torch_scatter
from typing import Dict, Optional, Sequence, Set, Tuple, Union
import tracemalloc
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.common.dist_utils import is_main_process
import glob
import random

class ThreeDReferDataset(BaseDataset):
    def __init__(
        self,
        text_processor,
        pts_root,
        ann_paths,
        question_type=None,
        filter_missing_gt_in_pth=False,
        pth_rel_subdir="scannetpp",
        eval_scene_ids=None,
        eval_scene_allowlist_file=None,
        eval_max_samples=None,
        instance_id_cache_file=None,
        instance_id_cache_write=False,
        write_filtered_annotations_to=None,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(text_processor, pts_root, ann_paths)
        self.pth_rel_subdir = pth_rel_subdir
        self.eval_max_samples = eval_max_samples
        self._eval_scene_set: Optional[Set[str]] = None
        self._pth_inst_cache: Dict[str, Optional[Set[int]]] = {}
        self._instance_id_cache_path: Optional[str] = None
        if instance_id_cache_file:
            p = str(instance_id_cache_file).strip().strip('"').strip("'")
            self._instance_id_cache_path = osp.abspath(osp.expanduser(p)) if p else None
        self._instance_id_cache_write = bool(instance_id_cache_write)
        self._write_filtered_ann_path: Optional[str] = None
        if write_filtered_annotations_to:
            w = str(write_filtered_annotations_to).strip().strip('"').strip("'")
            self._write_filtered_ann_path = osp.abspath(osp.expanduser(w)) if w else None
        self.scene_ids = {}
        n = 0
        self.use_xyz = True
        self.mode = 4
        new_annotation = []
        for ann in self.annotation:
            try:
                img_id = ann["scene_id"]
                if img_id not in self.scene_ids.keys():
                    self.scene_ids[img_id] = n
                    n += 1
                new_annotation.append(ann)
            except:
                pass
        if 'train' in ann_paths[0]:
            self.prefix = 'train'
            self.training = True
        else:
            self.prefix = 'val'
            self.training = False
        self.with_label = True

        self.question_type = question_type
        self.annotation = new_annotation
        if filter_missing_gt_in_pth:
            uniq_scenes = {a.get("scene_id") for a in self.annotation if a.get("scene_id") is not None}
            cache_loaded = False
            if self._instance_id_cache_path:
                cache_loaded = self._try_load_instance_id_cache()
            if cache_loaded:
                logging.info(
                    "ThreeDReferDataset: filter_missing_gt_in_pth using instance-id cache %r "
                    "(%d scenes in cache; filtering without re-reading each .pth unless a scene is missing).",
                    self._instance_id_cache_path,
                    len([k for k, v in self._pth_inst_cache.items() if v is not None]),
                )
            else:
                logging.info(
                    "ThreeDReferDataset: filter_missing_gt_in_pth will torch.load up to %d unique scene .pth "
                    "files under %s (subdir=%s). One full read per scene to cache instance ids; on full train "
                    "or slow/network storage this often takes several minutes and runs only once at init. "
                    "To skip next time, set dataset_init.instance_id_cache_file and instance_id_cache_write: true once.",
                    len(uniq_scenes),
                    self.pts_root,
                    self.pth_rel_subdir,
                )
            before = len(self.annotation)
            self.annotation = [a for a in self.annotation if self._ann_has_target_in_pth(a)]
            logging.info(
                "ThreeDReferDataset: filter_missing_gt_in_pth kept %d / %d annotations (pts_root=%s, subdir=%s)",
                len(self.annotation),
                before,
                pts_root,
                pth_rel_subdir,
            )
            if len(self.annotation) == 0:
                raise RuntimeError(
                    "All annotations were filtered out (no object_id in sampled_instance_anno_id). "
                    "Check pth paths and dataset_init.pth_rel_subdir."
                )
            if self._instance_id_cache_write and self._instance_id_cache_path and is_main_process():
                self._write_instance_id_cache()
            elif self._instance_id_cache_write and not self._instance_id_cache_path and is_main_process():
                logging.warning(
                    "ThreeDReferDataset: instance_id_cache_write is true but instance_id_cache_file is unset; not writing cache."
                )

        scene_allow = self._build_eval_scene_allowlist(eval_scene_ids, eval_scene_allowlist_file)
        if scene_allow:
            before_sc = len(self.annotation)
            ann_scenes = {a.get("scene_id") for a in self.annotation if a.get("scene_id") is not None}
            overlap = ann_scenes & scene_allow
            if not overlap:
                sample_ann = sorted(ann_scenes)[:12]
                sample_allow = sorted(scene_allow)[:12]
                raise RuntimeError(
                    "eval scene allowlist has no overlap with annotation scene_ids "
                    f"(allowlist n={len(scene_allow)}, ann scenes n={len(ann_scenes)}). "
                    f"Example ann scene_id: {sample_ann}. Example allowlist: {sample_allow}. "
                    "Use scene_id values from your val JSON (see scripts/trial_scenes.txt)."
                )
            self.annotation = [a for a in self.annotation if a.get("scene_id") in scene_allow]
            self._eval_scene_set = scene_allow
            logging.info(
                "ThreeDReferDataset: eval scene allowlist kept %d / %d annotations (%d scenes)",
                len(self.annotation),
                before_sc,
                len(scene_allow),
            )
            if len(self.annotation) == 0:
                raise RuntimeError(
                    "All annotations removed by eval_scene_ids / eval_scene_allowlist_file. "
                    "Check scene ids and that val JSON contains those scenes."
                )

        if self.eval_max_samples is not None and int(self.eval_max_samples) > 0:
            cap = int(self.eval_max_samples)
            if len(self.annotation) > cap:
                logging.info(
                    "ThreeDReferDataset: eval_max_samples=%d truncating %d -> %d",
                    cap,
                    len(self.annotation),
                    cap,
                )
                self.annotation = self.annotation[:cap]

        if self._write_filtered_ann_path and is_main_process():
            wp = self._write_filtered_ann_path
            d = osp.dirname(wp)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(wp, "w", encoding="utf-8") as f:
                json.dump(self.annotation, f, ensure_ascii=False)
            logging.info(
                "ThreeDReferDataset: wrote %d filtered annotations to %r (for faster re-runs, point train JSON here).",
                len(self.annotation),
                wp,
            )

        self.sp_filenames = self.get_sp_filenames()
        self.short_question_list = QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.with_elastic = False
        self.aug = True      
        self.data_cache = {}

    def _instance_ids_in_pth(self, scene_id: str) -> Optional[Set[int]]:
        if scene_id in self._pth_inst_cache:
            return self._pth_inst_cache[scene_id]
        pth = osp.join(self.pts_root, self.pth_rel_subdir, f"{scene_id}.pth")
        if not osp.isfile(pth):
            self._pth_inst_cache[scene_id] = None
            return None
        try:
            d = torch.load(pth, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(pth, map_location="cpu")
        inst = np.asarray(d["sampled_instance_anno_id"]).astype(np.int64).reshape(-1)
        u = {int(x) for x in np.unique(inst).tolist() if int(x) != -100}
        self._pth_inst_cache[scene_id] = u
        return u

    def _ann_has_target_in_pth(self, ann) -> bool:
        u = self._instance_ids_in_pth(ann["scene_id"])
        if u is None:
            return True
        oid = ann["object_id"]
        ids = [int(x) for x in oid] if isinstance(oid, list) else [int(oid)]
        return any(i in u for i in ids)

    _INSTANCE_CACHE_VERSION = "reason3d_instance_id_cache_v1"

    def _try_load_instance_id_cache(self) -> bool:
        path = self._instance_id_cache_path
        if not path or not osp.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception as e:
            logging.warning("ThreeDReferDataset: could not read instance_id_cache_file %r: %s", path, e)
            return False
        if blob.get("format") != self._INSTANCE_CACHE_VERSION:
            logging.warning("ThreeDReferDataset: instance cache %r has wrong format key; ignoring.", path)
            return False
        if osp.abspath(blob.get("pts_root", "")) != osp.abspath(self.pts_root):
            logging.warning(
                "ThreeDReferDataset: instance cache pts_root mismatch (cache=%r current=%r); ignoring.",
                blob.get("pts_root"),
                self.pts_root,
            )
            return False
        if str(blob.get("pth_rel_subdir", "")) != str(self.pth_rel_subdir):
            logging.warning(
                "ThreeDReferDataset: instance cache pth_rel_subdir mismatch (cache=%r current=%r); ignoring.",
                blob.get("pth_rel_subdir"),
                self.pth_rel_subdir,
            )
            return False
        inst = blob.get("instance_sets") or {}
        missing = set(blob.get("missing_pth") or [])
        for sid, ids in inst.items():
            if not isinstance(ids, list):
                continue
            self._pth_inst_cache[str(sid)] = {int(x) for x in ids}
        for sid in missing:
            self._pth_inst_cache[str(sid)] = None
        logging.info(
            "ThreeDReferDataset: loaded instance-id cache from %r (%d with ids, %d missing .pth).",
            path,
            len(inst),
            len(missing),
        )
        return True

    def _write_instance_id_cache(self) -> None:
        path = self._instance_id_cache_path
        assert path
        inst_out = {}
        missing_out = []
        for sid, u in self._pth_inst_cache.items():
            if u is None:
                missing_out.append(sid)
            else:
                inst_out[sid] = sorted(u)
        merged_inst = dict(inst_out)
        merged_miss = set(missing_out)
        if osp.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("format") == self._INSTANCE_CACHE_VERSION:
                    for sid, ids in (old.get("instance_sets") or {}).items():
                        if sid not in merged_inst and isinstance(ids, list):
                            merged_inst[sid] = ids
                    merged_miss.update(old.get("missing_pth") or [])
            except Exception as e:
                logging.warning("ThreeDReferDataset: could not merge existing cache %r: %s", path, e)
        blob = {
            "format": self._INSTANCE_CACHE_VERSION,
            "pts_root": osp.abspath(self.pts_root),
            "pth_rel_subdir": str(self.pth_rel_subdir),
            "instance_sets": {k: merged_inst[k] for k in sorted(merged_inst.keys())},
            "missing_pth": sorted(merged_miss),
        }
        d = osp.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=0)
        os.replace(tmp, path)
        logging.info(
            "ThreeDReferDataset: wrote instance-id cache to %r (%d scenes with ids, %d missing .pth).",
            path,
            len(merged_inst),
            len(merged_miss),
        )

    @staticmethod
    def _resolve_scene_list_path(path: str) -> str:
        if osp.isfile(path):
            return osp.abspath(path)
        try:
            from lavis.common.registry import registry

            repo = registry.get_path("repo_root")
            candidate = osp.normpath(osp.join(repo, path))
            if osp.isfile(candidate):
                return candidate
        except Exception:
            pass
        candidate2 = osp.normpath(osp.join(os.getcwd(), path))
        if osp.isfile(candidate2):
            return candidate2
        return path

    def _build_eval_scene_allowlist(
        self,
        eval_scene_ids: Optional[Union[Sequence[str], str]],
        eval_scene_allowlist_file: Optional[str],
    ) -> Optional[Set[str]]:
        ids: Set[str] = set()
        if eval_scene_ids is not None:
            if isinstance(eval_scene_ids, str):
                for part in eval_scene_ids.replace(",", " ").split():
                    part = part.strip()
                    if part:
                        ids.add(part)
            else:
                for x in eval_scene_ids:
                    if x is not None and str(x).strip():
                        ids.add(str(x).strip())
        if eval_scene_allowlist_file:
            fp = self._resolve_scene_list_path(str(eval_scene_allowlist_file).strip())
            if not osp.isfile(fp):
                raise FileNotFoundError(
                    f"eval_scene_allowlist_file not found: {eval_scene_allowlist_file} (resolved: {fp})"
                )
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    ids.add(s.split()[0])
        return ids if ids else None

    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.pts_root, self.pth_rel_subdir, "*" + ".pth"))
        assert len(filenames) > 0, "Empty dataset."
        filenames = sorted(filenames)
        if self._eval_scene_set:
            base = osp.join(self.pts_root, self.pth_rel_subdir)
            keep = {
                osp.join(base, f"{sid}.pth")
                for sid in self._eval_scene_set
                if osp.isfile(osp.join(base, f"{sid}.pth"))
            }
            filenames = sorted(keep)
            assert len(filenames) > 0, (
                "Empty dataset after eval scene allowlist: no matching .pth under "
                f"{base}. Check scene ids and pth_rel_subdir."
            )
        return filenames
        
    def load(self, filename):
        if self.with_label:
            #print(filename)
            data = torch.load(filename,weights_only=False)
            xyz, rgb, superpoint, semantic_label, instance_label = data['sampled_coords'],data['sampled_colors'],data['superpoints'],data['sampled_labels'],data['sampled_instance_anno_id']
            rgb = rgb / 0.5 - 1
            xyz = xyz[:, :3] - xyz[:, :3].mean(0)
            return xyz, rgb, superpoint, semantic_label, instance_label
        else:
            xyz, rgb, superpoint = torch.load(filename)
            rgb = rgb / 0.5 - 1
            xyz = xyz[:, :3] - xyz[:, :3].mean(0)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
    def transform_train(self,  xyz, rgb, superpoint, semantic_label, instance_label):
        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * 50 
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        valid_idxs = xyz.min(1) >= 0
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * 50
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        #print(valid_idxs.shape)
        #print(superpoint.shape)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        #if flip:
        #    m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(self, instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        # j = 0
        # while j < instance_label.max():
        #     if len(np.where(instance_label == j)[0]) == 0:
        #         instance_label[instance_label == instance_label.max()] = j
        #     j += 1
        return instance_label
    
    def get_ref_mask(self, instance_label, superpoint, object_id):
        if type(object_id) == list:
            ref_lbl = torch.isin(instance_label, torch.tensor(object_id))
        else:
            ref_lbl = instance_label == object_id
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask

    def __getitem__(self, index: int) -> Tuple:
        data = self.annotation[index]
        scan_id = data["scene_id"]
        if type(data["object_id"]) == list:
            object_id = [int(x) for x in data["object_id"]]
            ann_id = 0
        else:
            object_id = int(data["object_id"])
            ann_id = int(data["ann_id"])
        description = self.text_processor(data["description"])
        question_template = random.choice(self.short_question_list)
        question = question_template.format(description=description)
        # load point cloud
        #print(self.sp_filenames)
        
        for fn in self.sp_filenames:
            #print(fn)
            if scan_id in fn:
                sp_filename = fn
                break
        #print(scan_id)
        data = self.load(sp_filename)
        # if sp_filename in self.data_cache:
        #     data = self.data_cache[sp_filename]
        # else:
        #     data = self.load(sp_filename)
        #     if len(self.data_cache) < 50:
        #         self.data_cache[sp_filename] = data
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = data

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        #print(coord.shape)
        #print(superpoint.unique().max())
        semantic_label = torch.from_numpy(semantic_label).long()
        instance_label = torch.from_numpy(instance_label).long()
        #print(object_id)
        #print(instance_label.unique())
        gt_pmask, gt_spmask = self.get_ref_mask(instance_label, superpoint, object_id)
        answers = [random.choice(self.answer_list)]
        
        # if gt_pmask.int().max() != 1:
        #     #DUBUG
        #     print(np.unique(gt_pmask), torch.unique(instance_label), sp_filename, object_id, self.annotation[index]["scene_id"])
        #     data = self.load(sp_filename)
        #     inst = self.transform_test(*data)[-1]
        #     print(np.unique(inst))
        #     exit()

        assert gt_pmask.int().max() == 1, (
            f"No GT foreground for scene_id={scan_id!r} object_id={object_id!r} "
            f"(ann_id={ann_id!r}); target missing from sampled_instance_anno_id in {sp_filename!r}. "
            "Set datasets.3d_refer.dataset_init.filter_missing_gt_in_pth: true and check pth_rel_subdir."
        )
        return {
            'ann_ids': ann_id,
            'scan_ids': scan_id,
            'coord': coord,
            'coord_float': coord_float,
            'feat': feat,
            'superpoint': superpoint,
            'object_id': object_id,
            'gt_pmask': gt_pmask,
            'gt_spmask': gt_spmask,
            'sp_ref_mask': None,
            'lang_tokens': None,
            'answers': answers,
            "text_input": question,
        }

        return ann_id, scan_id, coord, coord_float, feat, superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask, lang_tokens
    
    def collater(self, batch):
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_words, answerss, text_input_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        batch_offsets = [0]
        n_answers = []
        superpoint_bias = 0

        for i, data in enumerate(batch):
            ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask, lang_tokens, answers, captions = list(data.values())
            
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            ann_ids.append(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            
            object_ids.append(object_id)
            
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            answerss.extend(answers)
            text_input_list.append(captions)

            n_answers.append(len(answers))

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_ids': object_ids,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            "answer": answerss,
            "text_input": text_input_list,
            'lang_tokenss': None,
            'lang_masks': None,
            'n_answers': torch.LongTensor(n_answers),
        }

    def __len__(self):
        return len(self.annotation)

QUESTION_LIST = [
    "Please segment the object according to the given 3D scene and the description: {description}.",
    "Given the 3D scene, segment this object according to the description: {description}.",
    "Respond the segmentation mask of the object: {description}.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]
