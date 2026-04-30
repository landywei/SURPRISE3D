#!/usr/bin/env python3
"""
Architecture narrative: Reason3D + Chain v3 (loss-only round)
=============================================================

Implementation reference:
  ``lavis.datasets.datasets.threedrefer_datasets_chainv3.ThreeDReferDatasetChainV3``
  ``lavis.datasets.builders.seg3d_builder_chainv3`` (builder id: ``3d_refer_chainv3``)
  ``lavis.models.reason3d_models.seg_loss_v3.CriterionV3``
  ``lavis.models.reason3d_models.reason3d_t5_chainv3.Reason3DT5ChainV3``
  ``lavis.tasks.refer_seg_task_v3.ThreeDReferSegTaskV3`` (task id: ``3d_refer_seg_v3``)

Companion design doc:
  ``Models/reason3d/docs/chainv3_design_proposal.md``  (loss derivations §2)
  ``Models/reason3d/docs/chainv3_literature_sweep.md`` (background)

What chain v3 changes (and what it does *not*)
----------------------------------------------
**Architecture is unchanged.** Same backbone, Q-Former, Flan-T5, ``[SEG]``
hidden, ``text_hidden_fcs``, and ``MaskDecoder`` as bare / chain v2 (see
``architecture_reason3d_baseline.py`` / ``architecture_reason3d_chain.py``).
``Reason3DT5ChainV3`` is a subclass of ``Reason3DT5`` whose **only** change
is the segmentation criterion and a tiny adapter that lets the loss see
per-instance GT masks at training time.

What chain v3 changes is the **loss** and the **per-sample GT it consumes**:

1. **Per-instance GT in the dataset.** ``ThreeDReferDatasetChainV3``
   subclasses ``ThreeDReferDatasetChain`` (so chain text targets — *"The
   answer is the chair. [SEG]."*-style — are preserved). Its overridden
   ``get_ref_mask`` builds, in addition to the union mask, a *list* of
   per-instance point masks and per-instance superpoint masks; its
   collater exposes them as ``gt_pmasks_per_instance`` and
   ``gt_spmasks_per_instance``.

2. **Best-of-set BCE+Dice in ``CriterionV3``.** When per-instance GT lists
   are present, ``CriterionV3`` replaces the union BCE+Dice with
   ``min_k (BCE+Dice against m_k)`` plus a one-sided coverage hinge
   ``lambda_cov * (1 - |pred|/|union|)_+``. This stops the loss from
   penalizing a model that predicts *one* valid chair when the query
   refers to *multiple* chairs.

3. **Scale-aware dice.** A size-normalized dice ``(1/sqrt(|m|+eps)) * Dice``
   is averaged across the batch and added with weight ``loss_weight[1] * 0.5``
   on top of the best-of-set / vanilla dice. Small-instance gradients no
   longer drown under large-instance ones.

4. **Optional Lovasz boundary** (``enable_boundary``) and
   **optional focal-BCE point-aux** (``enable_point_aux``), gated on the
   small-instance subset (``small_size_threshold``). Both are computed at
   point granularity by upsampling SP logits via ``samples["superpoints"]``
   and ``samples["batch_offsets"]``. Both are off by default in the
   shipped chain v3 YAML — they are wired but require explicit ablation.

5. **Per-instance hit@tau metric** in ``ThreeDReferSegTaskV3``. Reports
   ``max_k IoU(pred, m_k)`` and the corresponding hit rates at 0.25 / 0.50
   alongside the existing union mIoU / Acc@0.25 / Acc@0.50.

What stays the same (keeps chain v2 results comparable)
-------------------------------------------------------
* Same SPFormer / point encoder / Q-Former / Flan-T5 / mask decoder.
* Same ``predict_seg`` (one ``[SEG]`` per generation, one mask out).
* Same ``score_loss`` and ``aux_outputs`` recursion through intermediate
  mask-decoder layers.
* Same chain answer templates (``"The answer is the {name}. [SEG]."`` etc.)
  via the inherited ``ThreeDReferDatasetChain``.
* When the v3 dataset is replaced by the chain v2 dataset (or any dataset
  that does not emit per-instance lists), ``CriterionV3`` falls back to the
  same union BCE+Dice that ``Criterion`` computes — modulo the optional
  scale-aware term, which the ablation can turn off.

ASCII schematic (loss-only delta)
---------------------------------

    ThreeDReferDatasetChainV3.__getitem__
        ├── (union mask)      gt_pmask, gt_spmask
        └── (new)             gt_pmask_per_inst, gt_spmask_per_inst

    Reason3DT5ChainV3.forward(samples)
        ├── (unchanged)       PointExtractor → Q-Former → T5 → [SEG] → MaskDecoder
        └── self.criterion = _CriterionV3Adapter(CriterionV3, samples)
                              │
                              ▼
              CriterionV3(out, gt_pmasks, gt_spmasks, None,
                          gt_pmasks_per_instance=...,
                          gt_spmasks_per_instance=...,
                          superpoints=..., batch_offsets=...)
                              │
                              ▼
                  L = best_of_set(BCE+Dice) + score
                      + lambda_dice * 0.5 * scale_dice
                      [+ lambda_boundary * lovasz_small]
                      [+ lambda_point_aux * focal_pt_small]
                      + sum_l aux_layer_loss(l)

When to document chain v3 next to chain v2
------------------------------------------
* **Chain v2:** chain answer templates trained against union BCE/Dice.
* **Chain v3:** chain answer templates trained against best-of-set BCE/Dice
  with size-normalized dice and (opt-in) point-level small-object terms.
  Per-instance hit@tau is reported alongside the union metrics.

Run patterns
------------
Train::

    REASON3D_INIT_CKPT=/path/to/reason3d.pth bash scripts/run_surprise_finetune_chainv3.sh

Evaluate (full Surprise val with v3 metrics)::

    CFG=lavis/projects/reason3d/val/reason3d_surprise_zeroshot_chainv3.yaml \
      REASON3D_CKPT=/path/to/checkpoint.pth bash scripts/run_surprise_zeroshot_eval.sh
"""

if __name__ == "__main__":
    print(__doc__)
