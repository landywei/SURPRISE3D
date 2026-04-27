"""
Builder for chain-style answers (oracle object name before ``[SEG]``).

Baseline ``seg3d_builder.py`` is unchanged.
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.threedrefer_datasets_chain import ThreeDReferDatasetChain


@registry.register_builder("3d_refer_chain")
class ThreeDReferChainBuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDReferDatasetChain
    eval_dataset_cls = ThreeDReferDatasetChain

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dseg/defaults_chain.yaml"}
