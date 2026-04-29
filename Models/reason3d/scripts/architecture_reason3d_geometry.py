#!/usr/bin/env python3
"""
Architecture narrative: Reason3DT5Geo + GeoRelationalModule (geometry refinement)
=================================================================================

Implementation reference:
  ``lavis.models.reason3d_geo.reason3d_t5_geo.Reason3DT5Geo``
  ``lavis.models.reason3d_geo.geo_relational.GeoRelationalModule``
  ``lavis.models.reason3d_geo.geo_relational._GeoRelLayer``

Relation to baseline
--------------------
``Reason3DT5Geo`` subclasses ``Reason3DT5`` and **reuses the entire pipeline**
above. The only architectural insertion is **after** ``text_features`` are
produced from ``[SEG]`` and **before** ``MaskDecoder``: superpoint features
``sp_feats`` are replaced by ``sp_feats + delta`` where ``delta`` comes from
``GeoRelationalModule``. Batches must include ``coords_float`` (per-point 3D
coordinates aligned with ``superpoints`` indices) — typically via dataset
builder ``3d_refer_geo``.

Purpose
-------
The baseline path fuses language with superpoint tokens **without an explicit
3D neighborhood graph** on the decoder side. ``GeoRelationalModule`` adds a
**short, depth-limited graph neural refinement**: each superpoint may aggregate
evidence from **geometric k-nearest neighbors** on **instance-agnostic superpoint
centroids**, while edge messages are **conditioned on the same ``[SEG]`` vector**
that drives the mask decoder. Intuition: relational phrases depend on **who is
next to whom** in the scan; kNN message passing makes local layout explicit in
``sp_feats`` right before dense mask prediction.

GeoRelationalModule — internal algorithm
----------------------------------------
Per scene (between ``batch_offsets[i]`` and ``batch_offsets[i+1]``):

1. **Centroids**  
   For each superpoint id, average ``coords_float`` over constituent points
   (``scatter_mean``) → centroid cloud ``c`` of shape ``(M, 3)``.

2. **Coordinate normalization (graph construction only)**  
   Center ``c`` by its mean; divide each axis by std (clamped) → ``c_norm``.
   kNN is Euclidean in ``c_norm`` space (scale-invariant layout for neighbor
   choice).

3. **Sparse directed kNN graph**  
   For each of ``M`` nodes, keep **at most ``knn_k``** neighbors (default 16),
   excluding self, via chunked ``torch.cdist`` for memory. **Not** a complete
   graph: **O(M·k)** edges per scene.

4. **Fixed geometric edge features (4-D per directed neighbor)**  
   For edge ``i ← j``: displacement ``c_norm[j] - c_norm[i]`` (3) and
   ``log(1 + ||·||)`` (1). Passed through ``edge_mlp`` → ``edge_h`` of width
   ``hidden_dim``.

5. **Node initialization**  
   ``node_in``: ``sp_feats`` (dimension ``in_dim`` = point encoder ``media``)
   projected to ``hidden_dim``.  
   **Conditioning:** global ``cond`` is one row per scene — the **projected**
   ``[SEG]`` embedding (``cond_dim`` → ``hidden_dim``); broadcast to all ``M``
   nodes as ``cond_row``.

6. **Relational layers (× ``num_layers``)**  
   Each ``_GeoRelLayer``: pre-LayerNorm on node features; for each node, build
   messages from concatenation **[ h_i, h_j, edge_h_ij, cond ]**; MLP → message
   vectors; **learnable-temperature softmax attention over neighbors k** to
   aggregate messages; residual **h ← h + agg** (dropout optional). This is
   **attentional message passing** on the kNN graph, **language-conditioned**
   at every edge aggregation.

7. **Output residual on original feature space**  
   Final LayerNorm + linear back to ``in_dim``; **tanh** scaling; multiply by
   learnable **gamma** (small initial scale).  
   **Return:** ``sp_feats + delta_acc`` — a **bounded residual** so the geometry
   block perturbs the backbone representation rather than replacing it.

Training details (from code)
------------------------------
* ``use_checkpoint=True`` wraps each layer in **gradient checkpointing** during
  training to save activation memory on large scenes.

ASCII schematic (insertion point)::

    … ──► T5 decoder ──► [SEG] ──► text_hidden_fcs ──► text_features
                                                              │
    sp_feats (from PointExtractor) ───────────────────────────┼──┐
                                                              │  │
                                                              ▼  │
                                              GeoRelationalModule │
                                              (kNN + cond MP)       │
                                                              │  │
                                              sp_feats' ◄─────┘  │
                                                              │
                                                              ▼
                                                    MaskDecoder …

Note: If ``coords_float`` is absent, ``Reason3DT5Geo`` skips the module and
behaves like the baseline for that batch.
"""

if __name__ == "__main__":
    print(__doc__)
