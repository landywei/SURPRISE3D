"""
Chain v3 dataset: same chain-style answer text as ``ThreeDReferDatasetChain``,
plus **per-instance GT masks** for the best-of-set matching loss in
``CriterionV3`` and the per-instance hit@tau metric in
``ThreeDReferSegTaskV3``.

Implementation note
-------------------
We avoid duplicating ``ThreeDReferDataset.__getitem__`` (which would mean
re-loading and re-transforming the .pth a second time per item, doubling I/O).
Instead, we override ``get_ref_mask`` to compute per-instance masks alongside
the union mask and stash them on ``self._last_per_inst``; ``__getitem__`` then
calls ``super()`` (which calls our ``get_ref_mask`` once) and reads that
buffer. PyTorch ``DataLoader`` workers each instantiate their own dataset and
run ``__getitem__`` sequentially, so there is no concurrency issue.

The collater pops the new per-instance lists from each item dict before
delegating to ``super().collater(batch)`` (which positionally unpacks the
parent's fixed dict shape) and re-adds the new lists to the resulting batch
dict as ``gt_pmasks_per_instance`` and ``gt_spmasks_per_instance``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch_scatter

from lavis.datasets.datasets.threedrefer_datasets_chain import ThreeDReferDatasetChain


class ThreeDReferDatasetChainV3(ThreeDReferDatasetChain):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_per_inst: Dict[str, List[torch.Tensor]] = {"pms": [], "sms": []}

    def get_ref_mask(self, instance_label, superpoint, object_id):
        gt_pmask, gt_spmask = super().get_ref_mask(instance_label, superpoint, object_id)
        ids = (
            [int(x) for x in object_id]
            if isinstance(object_id, list)
            else [int(object_id)]
        )
        pms: List[torch.Tensor] = []
        sms: List[torch.Tensor] = []
        for oid in ids:
            ref_lbl = instance_label == oid
            pm = ref_lbl.float()
            if pm.max().item() < 1.0:
                # Instance id missing from sampled cloud (see filter_missing_gt_in_pth);
                # skip it from the per-instance set.
                continue
            sm = torch_scatter.scatter_mean(pm, superpoint, dim=-1)
            sm = (sm > 0.5).float()
            pms.append(pm)
            sms.append(sm)
        if not pms:
            # Defensive: keep at least one entry so best-of-set has a target.
            pms = [gt_pmask]
            sms = [gt_spmask]
        self._last_per_inst = {"pms": pms, "sms": sms}
        return gt_pmask, gt_spmask

    def __getitem__(self, index: int):
        out = super().__getitem__(index)
        out["gt_pmask_per_inst"] = list(self._last_per_inst["pms"])
        out["gt_spmask_per_inst"] = list(self._last_per_inst["sms"])
        return out

    def collater(self, batch):
        # Pop v3-only fields BEFORE super().collater so the parent's positional
        # ``list(data.values())`` unpack still matches the legacy 13-key dict shape.
        gt_pmasks_per_instance: List[List[torch.Tensor]] = [
            d.pop("gt_pmask_per_inst", []) for d in batch
        ]
        gt_spmasks_per_instance: List[List[torch.Tensor]] = [
            d.pop("gt_spmask_per_inst", []) for d in batch
        ]
        out = super().collater(batch)
        out["gt_pmasks_per_instance"] = gt_pmasks_per_instance
        out["gt_spmasks_per_instance"] = gt_spmasks_per_instance
        return out
