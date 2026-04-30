"""
Builder for chain v3 (chain-style answers + per-instance GT masks for best-of-set
matching loss in ``CriterionV3``).

Baseline ``seg3d_builder.py`` and chain v2 ``seg3d_builder_chain.py`` are unchanged.
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.threedrefer_datasets_chainv3 import ThreeDReferDatasetChainV3


@registry.register_builder("3d_refer_chainv3")
class ThreeDReferChainV3Builder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDReferDatasetChainV3
    eval_dataset_cls = ThreeDReferDatasetChainV3

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dseg/defaults_chainv3.yaml"}
