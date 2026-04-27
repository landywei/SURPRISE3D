flowchart TB
  subgraph inputs
    PC["Point cloud\n(XYZ+RGB per point)"]
    TXT["Text query q"]
  end

  subgraph encoder3d["3D encoder (PointExtractor)"]
    V["Voxelize / UNet"]
    SP["Pool → superpoints\nsp_feats ∈ ℝ^(M×32)"]
    PC --> V --> SP
  end

  subgraph bridge["Q-Former + T5"]
    AD["pc_adapter\n32 → 1408"]
    QF["Q-Former\n32 queries × superpoint tokens"]
    T5E["T5 encoder\n(visual tokens ∥ text tokens)"]
    T5D["T5 decoder\n(train: teacher forced)"]
    SEG["Hidden state at\n[SEG] positions"]
    TF["text_hidden_fcs\n→ text_features\n∈ ℝ^(B×1×512)"]
    SP --> AD --> QF --> T5E
    TXT --> T5E
    T5E --> T5D --> SEG --> TF
  end

  subgraph dec["MaskDecoder (unchanged)"]
    LP["lang_proj\n512 → 256 = query"]
    IP["input_proj(sp_feats)\n→ inst_feats"]
    MF["x_mask(sp_feats)\n→ mask_feats"]
    CA["Cross-attn:\nquery ↔ inst_feats"]
    HEAD["Mask logits per\nsuperpoint"]
    TF --> LP
    SP --> IP
    SP --> MF
    LP --> CA
    IP --> CA
    MF --> HEAD
  end

  SP -.->|"same tensor"| IP
  SP -.->|"same tensor"| MF
