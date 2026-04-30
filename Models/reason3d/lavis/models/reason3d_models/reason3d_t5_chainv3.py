"""
Reason3D + chain v3 loss extension (architecture is identical to ``Reason3DT5``).

Only two things change versus the canonical model:

1. ``self.criterion`` is a ``CriterionV3`` (best-of-set + scale-aware dice +
   optional Lovasz boundary + optional focal-BCE point-aux).
2. The single ``self.criterion(...)`` call inside ``Reason3DT5.forward`` is
   transparently augmented to pass per-instance GT lists and the
   ``superpoints`` / ``batch_offsets`` tensors that ``CriterionV3`` consumes.

We avoid copy-pasting the parent ``forward`` body by stashing the v3-only
extras on ``self.criterion._pending_extras`` for the duration of one forward
pass; ``CriterionV3.forward`` reads from that buffer when its kwargs are not
supplied at the legacy four-positional-arg call site.

We deliberately do **not** replace ``self.criterion`` with a plain Python
wrapper: ``Criterion`` is a ``nn.Module`` (registered via ``gorilla.LOSSES``)
and PyTorch's ``__setattr__`` rejects assigning a non-Module to a name that
already names a child module.

``predict_seg`` is inherited unchanged.
"""

from __future__ import annotations

from lavis.common.registry import registry
from lavis.models.reason3d_models.reason3d_t5 import Reason3DT5
from lavis.models.reason3d_models.seg_loss_v3 import CriterionV3


_LEGACY_CRITERION_KEYS = ("loss_weight", "loss_fun")


@registry.register_model("reason3d_t5_chainv3")
class Reason3DT5ChainV3(Reason3DT5):
    """Same module graph as ``Reason3DT5``; uses ``CriterionV3``.

    The model accepts the same ``seg_criterion_cfg`` shape as the base, plus
    v3 keys (e.g. ``enable_best_of_set``, ``enable_scale_aware``,
    ``enable_boundary``, ``enable_point_aux``, ``lambda_boundary``,
    ``lambda_point_aux``, ``lambda_cov``, ``small_size_threshold``,
    ``boundary_loss_type``). Legacy keys (``loss_weight``, ``loss_fun``)
    are passed both to the temporary base ``Criterion`` (built by
    ``super().__init__``) and to the final ``CriterionV3`` we install.
    """

    def __init__(self, **kwargs) -> None:
        raw_cfg = dict(kwargs.get("seg_criterion_cfg") or {})
        legacy_cfg = {k: raw_cfg[k] for k in _LEGACY_CRITERION_KEYS if k in raw_cfg}
        v3_extra = {k: v for k, v in raw_cfg.items() if k not in _LEGACY_CRITERION_KEYS}
        # Hand the base its legacy-only criterion config; we then replace
        # the registered child module ``criterion`` with the v3 instance.
        kwargs = dict(kwargs)
        kwargs["seg_criterion_cfg"] = legacy_cfg
        super().__init__(**kwargs)
        # Replacing one ``nn.Module`` child with another is allowed by
        # ``__setattr__`` (PyTorch swaps the entry in ``self._modules``).
        self.criterion = CriterionV3(**legacy_cfg, **v3_extra)

    def forward(self, samples):
        # Stash per-step extras on the criterion module so the inherited
        # ``Reason3DT5.forward`` does not need to know about v3 kwargs.
        self.criterion._pending_extras = {
            "gt_pmasks_per_instance": samples.get("gt_pmasks_per_instance"),
            "gt_spmasks_per_instance": samples.get("gt_spmasks_per_instance"),
            "superpoints": samples.get("superpoints"),
            "batch_offsets": samples.get("batch_offsets"),
        }
        try:
            return super().forward(samples)
        finally:
            self.criterion._pending_extras = None
