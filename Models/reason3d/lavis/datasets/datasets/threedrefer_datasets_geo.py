"""
Same as ``ThreeDReferDataset`` but collates ``coords_float`` for geometry-aware models.

Baseline ``threedrefer_datasets.py`` is unchanged.
"""

from __future__ import annotations

import torch

from lavis.datasets.datasets.threedrefer_datasets import ThreeDReferDataset


class ThreeDReferDatasetGeo(ThreeDReferDataset):
    def collater(self, batch):
        coords_float = torch.cat([d["coord_float"] for d in batch], dim=0)
        out = super().collater(batch)
        out["coords_float"] = coords_float
        return out
