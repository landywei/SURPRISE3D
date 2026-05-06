"""
Builder for chain v3 CoT (multi-step reasoning, P4 CoT answer template
for landmark-relational queries + per-instance GT masks).

Inherits all chain v3 behavior; only the dataset class changes.
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.threedrefer_datasets_chainv3_cot import (
    ThreeDReferDatasetChainV3CoT,
)


@registry.register_builder("3d_refer_chainv3_cot")
class ThreeDReferChainV3CoTBuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDReferDatasetChainV3CoT
    eval_dataset_cls = ThreeDReferDatasetChainV3CoT

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dseg/defaults_chainv3_cot.yaml"}
