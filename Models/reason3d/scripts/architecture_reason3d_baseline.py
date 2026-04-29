#!/usr/bin/env python3
"""
Architecture narrative: Reason3DT5 (baseline 3D referring segmentation)
=======================================================================

Implementation reference:
  ``lavis.models.reason3d_models.reason3d_t5.Reason3DT5``
  ``lavis.models.reason3d_models.mask_decoder.MaskDecoder``
  ``lavis.models.reason3d_models.point_extractor.PointExtractor``

Overview
--------
Reason3DT5 is a **encoder–fusion–language–decoder** stack for **language-conditioned
3D instance segmentation** on superpoints. The scene is encoded once into
superpoint features; a frozen (mostly) **Flan-T5** reasons over the question and
produces a compact **segmentation query** aligned to a special ``[SEG]`` token;
a **transformer mask decoder** then predicts per-superpoint logits from that
query and the (optionally refined) superpoint features.

Data flow (training ``forward`` / inference ``predict_seg``)
-------------------------------------------------------------
1. **PointExtractor (3D backbone)**  
   Raw batched point clouds (coordinates, colors, etc.) are mapped to a tensor
   ``sp_feats`` of shape ``(total_superpoints, media)`` — one feature vector per
   superpoint, with ``batch_offsets`` delimiting scenes.

2. **Batching for attention**  
   ``MaskDecoder.get_batches`` packs variable-length superpoint sequences into a
   fixed-width tensor ``(B, max_sp, media)`` plus a boolean padding mask so
   subsequent transformers see valid attention masks only on real superpoints.

3. **Vision–Q-Former bridge**  
   ``pc_adapter``: linear projection from ``media`` → 1408 for BLIP2-style
   Q-Former compatibility.  
   **Q-Former** (BERT backbone, query_tokens fixed): **learnable queries** attend
   **cross-attention** to superpoint tokens, producing a **fixed-length** visual
   summary per scene (``num_query_token`` vectors). This compresses the whole
   scan into a small set of language-ready visual tokens.

4. **T5 encoder fusion**  
   ``t5_proj`` maps Q-Former hidden size → T5 hidden size.  
   Encoder **input embeddings** are ``[ visual_tokens | text_token_embeddings ]``
   (concatenated along sequence). The **T5 encoder** therefore sees both the
   question (and optional prompt) and the Q-Former summary of the scene in one
   multimodal prefix.

5. **T5 decoder and ``[SEG]`` anchor**  
   The **decoder** is trained (teacher forcing) or run (greedy decode + one
   teacher-forced readout at inference) to generate an **answer string** that
   includes a dedicated ``[SEG]`` token. Hidden states at **``[SEG]`` positions**
   are pooled / selected and passed through ``text_hidden_fcs`` → a vector
   ``text_features`` of dimension ``d_text`` (e.g. 512). That vector is the
   **linguistic segmentation embedding**: “what to segment given the reasoning
   chain up to segmentation.”

6. **MaskDecoder**  
   Superpoint features are projected to ``d_model``; ``text_features`` are
   projected to the same space. A stack of **self-attention** on superpoints and
   **cross-attention** from superpoints to the language query refines geometric
   tokens under the current expression. A **bilinear/dot-product head**
   (``einsum`` over query and mask feature stream) yields **per-superpoint mask
   logits**; optional **mask-to-attention** gating can suppress padded regions.

7. **Loss**  
   **Language:** T5 cross-entropy on decoder tokens.  
   **Segmentation:** ``Criterion`` on predicted vs ground-truth point / superpoint
   masks (plus auxiliary terms as configured).

Design role of each block
-------------------------
* **PointExtractor**: geometric and semantic local evidence in 3D.  
* **Q-Former + T5 encoder**: global multimodal **reasoning** and alignment — the
  model can chain language before committing to a segmentation embedding.  
* **``[SEG]`` + text_hidden_fcs**: explicit **bottleneck** from free-form text to
  a single conditioning vector for the mask head.  
* **MaskDecoder**: **dense** fusion — every superpoint logit is informed by the
  final language-conditioned query.

ASCII schematic::

    Points ──► PointExtractor ──► sp_feats (total_sp × media)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       │
            get_batches + pc_adapter                         │
                    ▼                                       │
            Q-Former(queries ← sp_feats)                     │
                    ▼                                       │
            t5_proj ──► visual_tokens                       │
                    │                                       │
    text_input ──► token_embed ─────────────────────────────┤
                    │                                       │
                    └──► [ visual_tokens | text ] ──► T5 encoder
                                              │
                    answer / generate ──► T5 decoder
                                              │
                    [SEG] hidden ──► text_hidden_fcs ──► text_features
                                              │                │
                    sp_feats ─────────────────┴────────────────┘
                                              │
                                              ▼
                                    MaskDecoder ──► mask logits
                                              │
                                              ▼
                                    Criterion (seg) + CE (LM)
"""

if __name__ == "__main__":
    print(__doc__)
