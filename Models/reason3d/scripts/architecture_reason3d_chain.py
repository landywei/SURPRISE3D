#!/usr/bin/env python3
"""
Architecture narrative: Reason3D+Chain (chain-style supervision, same backbone as bare)
=========================================================================================

Implementation reference:
  ``lavis.datasets.datasets.threedrefer_datasets_chain.ThreeDReferDatasetChain``
  ``lavis.datasets.builders.seg3d_builder_chain`` (builder id: ``3d_refer_chain``)
  ``lavis.models.reason3d_models.reason3d_t5.Reason3DT5``  (same class as bare; no ``_geo`` subclass)

What “chain” is (and is not)
----------------------------
**Chain is not a third neural module** alongside the point encoder and mask decoder.
The **network architecture is identical** to baseline Reason3D: ``PointExtractor`` →
Q-Former → Flan-T5 encoder/decoder → ``[SEG]`` hidden state → ``text_hidden_fcs`` →
``MaskDecoder``. There is **no** ``GeoRelationalModule`` unless you combine configs
explicitly.

What changes is **supervision of the language head** during training (and thus the
**distribution of decoded text** at inference): the decoder is trained to emit a
**short explicit phrase** that names the ground-truth referent (from annotation
fields such as ``object_name``), **immediately followed** by the literal token
``[SEG]``, instead of being trained only on minimal ``[SEG]``-centric replies.

Dataset layer (only moving part at train time)
-----------------------------------------------
``ThreeDReferDatasetChain`` subclasses ``ThreeDReferDataset`` and overrides only
the **``answers``** field in ``__getitem__``. For each sample it:

1. Reads **oracle** name text from configurable keys (default ``object_name``),
   flattened to a human-readable phrase (comma-separated if multiple distinct
   tokens).

2. Wraps that phrase in a **template** drawn from ``CHAIN_ANSWER_LIST``, e.g.
   ``"The answer is the {name}. [SEG]."`` — always preserving the substring
   ``[SEG]`` so tokenization and the **same pooling rule** as bare (hidden states
   at ``[SEG]`` positions → ``text_hidden_fcs``) remain valid.

3. If no usable name is found: optionally falls back to the **plain** baseline
   answer list (``chain_answer_fallback_plain``) or a minimal ``"[SEG]."`` string.

So **chain = data conditioning**: the T5 cross-entropy loss teaches the model to
**verbalize a category / object hypothesis before grounding**, while the
**segmentation head loss** still trains on the same masks as the parent dataset.

Inference and evaluation
--------------------------
``predict_seg`` is **unchanged** in code path: greedy (or configured) decode of a
full answer string, then **re-read** decoder hidden states at ``[SEG]`` to form
``text_features`` for ``MaskDecoder``. The practical difference is that the
decoded prefix is often **interpretable** (names or short clauses) before
``[SEG]``, which helps error analysis; forks may add **repetition penalty** and
**no-repeat n-gram** constraints in generation kwargs to reduce degenerate loops
on longer targets (see project YAMLs / eval scripts).

When to document chain next to geo
----------------------------------
* **Geo:** adds ``GeoRelationalModule`` **inside** ``Reason3DT5Geo`` between
  ``[SEG]`` features and ``MaskDecoder``.
* **Chain:** same ``Reason3DT5`` module graph as bare; swap dataset builder to
  ``3d_refer_chain`` and train with chain targets.

ASCII schematic (data only)::

    Bare train target (conceptual):     … → "[SEG]."  (or short baseline templates)
    Chain train target (conceptual):    … → "The answer is the <oracle name>. [SEG]."
                                                    │
                                                    └── same [SEG] → text_hidden_fcs → MaskDecoder

"""

if __name__ == "__main__":
    print(__doc__)
