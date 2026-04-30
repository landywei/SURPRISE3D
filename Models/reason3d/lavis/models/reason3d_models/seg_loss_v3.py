"""
Chain v3 segmentation loss for Reason3D.

Adds, on top of the canonical ``Criterion`` (``seg_loss.py``):
- **best-of-set matching** for multi-target referents (problem 1 in
  ``Models/reason3d/docs/chainv3_design_proposal.md``): when multiple GT
  instances belong to the referent (e.g. *"the chairs"*), the BCE+Dice loss
  is taken as the *minimum* over the GT set, plus a one-sided coverage hinge
  so the model cannot trivially game the score by predicting the smallest
  instance;
- **scale-aware dice** (problem 2): per-sample dice is weighted by
  ``1 / sqrt(|m|+eps)`` so small instances contribute proportionally larger
  gradient than large ones;
- optional **Lovasz boundary** auxiliary (problem 2; opt-in) on the
  small-instance subset, applied at point granularity by upsampling SP
  logits via the ``superpoints`` index map;
- optional **focal-BCE point-level auxiliary** (problem 2; opt-in) on the
  small-instance subset, also at point granularity.

The canonical ``Criterion`` and ``Reason3DT5`` paths are not modified. With
``enable_best_of_set=False, enable_scale_aware=False, enable_boundary=False,
enable_point_aux=False``, ``CriterionV3`` is numerically identical to
``Criterion`` (same BCE+Dice on the union mask, same score loss, same
``aux_outputs`` recursion).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from lavis.models.reason3d_models.seg_loss import (
    Criterion,
    _pred_masks_scores_for_loss,
    dice_loss,
    get_iou,
    sigmoid_focal_loss,
)


# ---------------------------------------------------------------------------
# Lovasz hinge (Berman et al., CVPR 2018) — vendored compact implementation.
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovasz-hinge over a single-class flat tensor of valid positions."""
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


# ---------------------------------------------------------------------------
# CriterionV3
# ---------------------------------------------------------------------------

@gorilla.LOSSES.register_module()
class CriterionV3(Criterion):
    """Chain v3 segmentation loss.

    Hyper-parameters (all forward-compatible with the legacy ``Criterion``):

    Args:
        loss_weight: ``[bce, dice, score, sample]`` — same meaning as in
            ``Criterion``. Default mirrors current chain YAMLs
            ``[1.0, 1.0, 0.5, 5.0]``.
        loss_fun: ``'bce'`` or ``'focal'`` — only used when ``sp_ref_masks``
            is provided (legacy sample loss).
        enable_best_of_set: when ``True`` and per-instance GT lists are
            provided to ``forward``, replace the union BCE+Dice with the
            minimum-over-GT-set version (plus coverage hinge).
        enable_scale_aware: when ``True``, add a size-normalized dice term
            ``(1/sqrt(|m|)) * Dice`` averaged across the batch, weighted by
            ``loss_weight[1] * 0.5`` so the total dice contribution
            stays comparable to the legacy term.
        enable_boundary: when ``True`` (default ``False``), add a Lovasz
            hinge auxiliary on the *small-instance* subset, applied at
            point granularity (requires ``superpoints`` and
            ``batch_offsets`` in ``forward`` kwargs).
        enable_point_aux: when ``True`` (default ``False``), add a
            sigmoid-focal point-level auxiliary on the small-instance
            subset (best-of-set in point space).
        lambda_boundary, lambda_point_aux, lambda_cov: scalar weights for
            the boundary, point-aux, and coverage-hinge terms.
        small_size_threshold: instance is "small" iff its point-mass is
            strictly below this many points. Used to gate boundary and
            point-aux losses so big-instance gradient stays clean.
        boundary_loss_type: ``'lovasz'`` (only option for now). Hook for
            future ``'surface'`` / ``'tversky'`` variants.
    """

    def __init__(
        self,
        loss_weight: Sequence[float] = (1.0, 1.0, 0.5, 5.0),
        loss_fun: str = "focal",
        enable_best_of_set: bool = True,
        enable_scale_aware: bool = True,
        enable_boundary: bool = False,
        enable_point_aux: bool = False,
        lambda_boundary: float = 0.5,
        lambda_point_aux: float = 0.5,
        lambda_cov: float = 0.1,
        small_size_threshold: int = 50,
        boundary_loss_type: str = "lovasz",
    ) -> None:
        super().__init__(loss_weight=list(loss_weight), loss_fun=loss_fun)
        self.enable_best_of_set = bool(enable_best_of_set)
        self.enable_scale_aware = bool(enable_scale_aware)
        self.enable_boundary = bool(enable_boundary)
        self.enable_point_aux = bool(enable_point_aux)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_point_aux = float(lambda_point_aux)
        self.lambda_cov = float(lambda_cov)
        self.small_size_threshold = int(small_size_threshold)
        self.boundary_loss_type = str(boundary_loss_type)
        if self.boundary_loss_type not in {"lovasz"}:
            raise ValueError(
                f"CriterionV3.boundary_loss_type={boundary_loss_type!r} not supported (only 'lovasz' for now)."
            )
        # Per-step buffer set by Reason3DT5ChainV3.forward so the legacy
        # 4-positional-arg call site inside Reason3DT5.forward keeps working
        # without copy-pasting the parent forward. ``forward`` below merges
        # this buffer into its kwargs when they are not supplied directly.
        self._pending_extras: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _bce_dice_against(
        self,
        pred_logit: torch.Tensor,  # [M_max]
        pad: torch.Tensor,         # [M_max] float (1=valid)
        target: torch.Tensor,      # [M_max] float in {0,1}
    ) -> torch.Tensor:
        """Single-target BCE+Dice scalar with the standard ``loss_weight`` split."""
        bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
        bce = (bce * pad).sum() / pad.sum().clamp_min(1.0)
        inputs = pred_logit.sigmoid() * pad
        num = 2 * (inputs * target).sum()
        den = inputs.sum() + target.sum()
        dice = 1 - (num + 1) / (den + 1)
        return self.loss_weight[0] * bce + self.loss_weight[1] * dice

    def _best_of_set_per_sample(
        self,
        pred_logit: torch.Tensor,       # [M_max]
        pad: torch.Tensor,              # [M_max] float
        per_inst_sp: List[torch.Tensor],  # each [M_b]
        union_sp: torch.Tensor,         # [M_max] float
    ) -> torch.Tensor:
        """min_k (BCE+Dice) over per-instance SP masks, plus one-sided coverage hinge."""
        device = pred_logit.device
        M_max = pred_logit.shape[0]

        if not per_inst_sp:
            return self._bce_dice_against(pred_logit, pad, union_sp.float())

        losses = []
        for m_k in per_inst_sp:
            tgt = torch.zeros(M_max, dtype=torch.float32, device=device)
            mk = m_k.float().to(device)
            tgt[: mk.shape[0]] = mk
            losses.append(self._bce_dice_against(pred_logit, pad, tgt))
        best = torch.stack(losses).min()

        # Coverage hinge: one-sided penalty if pred mass is much smaller than union.
        # We never punish over-prediction; multi-target picking one referent is fine.
        union_mass = (union_sp.float() * pad).sum().clamp_min(1.0)
        pred_mass = (pred_logit.sigmoid() * pad).sum()
        cov = (1.0 - pred_mass / union_mass).clamp_min(0.0)
        return best + self.lambda_cov * cov

    @staticmethod
    def _per_sample_scaled_dice(
        pred_logits: torch.Tensor,  # [B, M]
        target: torch.Tensor,       # [B, M] in {0,1}
        pad_mask: torch.Tensor,     # [B, M] float
        eps: float = 1.0,
    ) -> torch.Tensor:
        """Size-normalized batch dice: weighted average with weight ``1/sqrt(|m|+eps)``."""
        inputs = pred_logits.sigmoid() * pad_mask
        target_v = target * pad_mask
        numerator = 2 * (inputs * target_v).sum(-1)
        denominator = inputs.sum(-1) + target_v.sum(-1)
        dice = 1 - (numerator + 1) / (denominator + 1)  # [B]
        sizes = target_v.sum(-1)
        weights = 1.0 / (sizes.sqrt() + eps)
        return (dice * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _local_pt_logits_for_sample(
        pred_masks: torch.Tensor,        # [B, M]
        b: int,
        superpoints: torch.Tensor,       # [N_total] long
        batch_offsets: torch.Tensor,     # [B+1]
    ) -> Optional[torch.Tensor]:
        """Upsample SP logits to point granularity for sample b (returns ``None`` if empty)."""
        start = int(batch_offsets[b].item())
        end = int(batch_offsets[b + 1].item())
        M_b = end - start
        if M_b <= 0:
            return None
        sp_logit_b = pred_masks[b, :M_b]
        in_b = (superpoints >= start) & (superpoints < end)
        if int(in_b.sum().item()) == 0:
            return None
        local_idx = (superpoints[in_b] - start).long().clamp_min(0).clamp_max(M_b - 1)
        return sp_logit_b[local_idx]

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        pred,
        gt_pmasks,
        gt_spmasks,
        sp_ref_masks=None,
        gt_pmasks_per_instance: Optional[Sequence[Sequence[torch.Tensor]]] = None,
        gt_spmasks_per_instance: Optional[Sequence[Sequence[torch.Tensor]]] = None,
        superpoints: Optional[torch.Tensor] = None,
        batch_offsets: Optional[torch.Tensor] = None,
    ):
        """Backward-compatible super-set of ``Criterion.forward``.

        New keyword args (optional):
            gt_pmasks_per_instance: list of length ``B``; entry ``b`` is a list
                of per-instance point masks ``[N_b]`` for sample ``b``.
            gt_spmasks_per_instance: same shape but at superpoint granularity.
            superpoints: global SP-index tensor over points (already in
                ``samples`` from the dataset collater).
            batch_offsets: ``[B+1]`` tensor of cumulative SP counts (already in
                ``samples``).

        When the four kwargs above are not supplied at the call site, we fall
        back to the per-step ``self._pending_extras`` buffer (set by
        ``Reason3DT5ChainV3.forward``). This lets the inherited
        ``Reason3DT5.forward`` keep its legacy four-positional-arg call
        ``self.criterion(out, gt_pmasks, gt_spmasks, None)`` unchanged.
        """
        if (
            gt_pmasks_per_instance is None
            and gt_spmasks_per_instance is None
            and superpoints is None
            and batch_offsets is None
            and self._pending_extras is not None
        ):
            pe = self._pending_extras
            gt_pmasks_per_instance = pe.get("gt_pmasks_per_instance")
            gt_spmasks_per_instance = pe.get("gt_spmasks_per_instance")
            superpoints = pe.get("superpoints")
            batch_offsets = pe.get("batch_offsets")

        loss_out = {}

        pred_masks, pred_scores = _pred_masks_scores_for_loss(pred["masks"], pred["scores"])
        pad_masks = ~pred["batch_mask"]
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        B, M_max = pred_masks.shape

        # ----- score loss (same as base) -----
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores > 0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(
                F.mse_loss(pred_scores, tgt_scores, reduction="none"), score_mask
            ).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_out["score_loss"] = score_loss

        # ----- legacy sample loss path (unchanged) -----
        if sp_ref_masks is not None:
            ref_padding = pad_sequence(sp_ref_masks, batch_first=True)
            ref_scores = pred["ref_scores"]
            if self.loss_fun == "focal":
                from lavis.models.reason3d_models.seg_loss import (
                    SigmoidFocalClassificationLoss,
                )

                sample_criterion = SigmoidFocalClassificationLoss()
                cls_weights = pad_masks.float()
                cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
                cls_weights /= torch.clamp(cls_normalizer, min=1.0)
                sample_loss = sample_criterion(
                    ref_scores.unsqueeze(-1), ref_padding.unsqueeze(-1).float(), weights=cls_weights
                )
                sample_loss = (sample_loss.squeeze(-1) * pad_masks).sum(-1)
                sample_loss = sample_loss.mean()
            elif self.loss_fun == "bce":
                sample_loss = F.binary_cross_entropy_with_logits(
                    ref_scores, ref_padding.float(), reduction="none"
                )
                sample_loss = (sample_loss * pad_masks).sum(-1) / pad_masks.sum(-1).clamp_min(1)
                sample_loss = sample_loss.mean()
            else:
                raise NotImplementedError
        else:
            sample_loss = None

        # ----- mask reference loss (best-of-set or vanilla union) -----
        if self.enable_best_of_set and gt_spmasks_per_instance is not None:
            per_sample_losses = []
            for b in range(B):
                per_inst_b = list(gt_spmasks_per_instance[b]) if gt_spmasks_per_instance[b] is not None else []
                per_sample_losses.append(
                    self._best_of_set_per_sample(
                        pred_masks[b], pad_masks[b].float(), per_inst_b, tgt_padding[b]
                    )
                )
            mask_ref_loss = torch.stack(per_sample_losses).mean()
            loss_out["mask_ref_loss"] = mask_ref_loss
        else:
            # Vanilla path: matches base ``Criterion.forward``.
            mask_bce_loss = F.binary_cross_entropy_with_logits(
                pred_masks, tgt_padding.float(), reduction="none"
            )
            mask_bce_loss = (mask_bce_loss * pad_masks).sum(-1) / pad_masks.sum(-1).clamp_min(1)
            mask_bce_loss = mask_bce_loss.mean()
            mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)
            mask_ref_loss = (
                self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss
            )
            loss_out["mask_bce_loss"] = mask_bce_loss
            loss_out["mask_dice_loss"] = mask_dice_loss

        # ----- scale-aware dice (additive) -----
        if self.enable_scale_aware:
            scale_dice = self._per_sample_scaled_dice(
                pred_masks, tgt_padding.float(), pad_masks.float()
            )
            loss_out["mask_scale_dice"] = scale_dice
            mask_ref_loss = mask_ref_loss + (self.loss_weight[1] * 0.5) * scale_dice

        # ----- boundary (Lovasz) on small-instance subset, point-level -----
        boundary_loss = pred_masks.new_zeros(())
        if (
            self.enable_boundary
            and gt_pmasks_per_instance is not None
            and superpoints is not None
            and batch_offsets is not None
        ):
            n_contrib = 0
            for b in range(B):
                inst = list(gt_pmasks_per_instance[b]) if gt_pmasks_per_instance[b] is not None else []
                small = [m for m in inst if int(m.sum().item()) < self.small_size_threshold]
                if not small:
                    continue
                pt_logits = self._local_pt_logits_for_sample(
                    pred_masks, b, superpoints, batch_offsets
                )
                if pt_logits is None:
                    continue
                # Build small-instance union as the boundary target. Lengths must match.
                tgt = torch.zeros_like(pt_logits, dtype=torch.float32)
                ok = False
                for m in small:
                    if m.shape[0] == tgt.shape[0]:
                        tgt = torch.maximum(tgt, m.float().to(tgt.device))
                        ok = True
                if not ok:
                    continue
                boundary_loss = boundary_loss + _lovasz_hinge_flat(pt_logits.float(), tgt)
                n_contrib += 1
            if n_contrib > 0:
                boundary_loss = boundary_loss / n_contrib
            loss_out["mask_boundary_loss"] = boundary_loss

        # ----- focal-BCE point-aux (best-of-set in point space) -----
        point_aux_loss = pred_masks.new_zeros(())
        if (
            self.enable_point_aux
            and gt_pmasks_per_instance is not None
            and superpoints is not None
            and batch_offsets is not None
        ):
            n_contrib = 0
            for b in range(B):
                inst = list(gt_pmasks_per_instance[b]) if gt_pmasks_per_instance[b] is not None else []
                small = [m for m in inst if int(m.sum().item()) < self.small_size_threshold]
                if not small:
                    continue
                pt_logits = self._local_pt_logits_for_sample(
                    pred_masks, b, superpoints, batch_offsets
                )
                if pt_logits is None:
                    continue
                ks = []
                for m in small:
                    if m.shape[0] != pt_logits.shape[0]:
                        continue
                    ks.append(sigmoid_focal_loss(pt_logits.float(), m.float().to(pt_logits.device)))
                if not ks:
                    continue
                point_aux_loss = point_aux_loss + torch.stack(ks).min()
                n_contrib += 1
            if n_contrib > 0:
                point_aux_loss = point_aux_loss / n_contrib
            loss_out["mask_point_aux_loss"] = point_aux_loss

        # ----- final scalar -----
        loss = mask_ref_loss + self.loss_weight[2] * score_loss
        if sample_loss is not None:
            loss = loss + self.loss_weight[3] * sample_loss
            loss_out["sample_loss"] = sample_loss
        if self.enable_boundary:
            loss = loss + self.lambda_boundary * boundary_loss
        if self.enable_point_aux:
            loss = loss + self.lambda_point_aux * point_aux_loss

        # ----- aux outputs from intermediate mask-decoder layers (legacy) -----
        if "aux_outputs" in pred:
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, pad_masks, gt_spmasks)
                loss = loss + loss_i
                loss_out.update(loss_out_i)

        loss_out["loss"] = loss
        return loss, loss_out
