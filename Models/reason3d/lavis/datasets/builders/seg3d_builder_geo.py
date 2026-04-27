"""
Builder for Surprise3D / 3D refer with ``coords_float`` in each batch (geometry path).

Baseline ``seg3d_builder.py`` is unchanged.
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.threedrefer_datasets_geo import ThreeDReferDatasetGeo


@registry.register_builder("3d_refer_geo")
class ThreeDReferGeoBuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDReferDatasetGeo
    eval_dataset_cls = ThreeDReferDatasetGeo

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dseg/defaults_geo.yaml"}
