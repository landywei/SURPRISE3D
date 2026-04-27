"""
Geometry-aware relational refinement on superpoint features (decoder-side).

Used by ``Reason3DT5Geo``; keeps baseline Reason3D code paths unchanged.

Design notes (v2):
  * Pre-LayerNorm on node features before message passing (more stable than post-LN here).
  * Learnable positive attention temperature (softer / sharper neighbor softmax).
  * Edge MLP with LayerNorm + GELU on fixed geometry features.
  * Final LayerNorm + tanh-bounded projection so residuals on ``sp_feats`` stay bounded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_mean


class _GeoRelLayer(nn.Module):
    """One relational block: pre-norm nodes, language in context, neighbor softmax attention."""

    def __init__(self, d_model: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        in_f = d_model * 2 + edge_dim + d_model
        self.norm_in = nn.LayerNorm(d_model)
        self.msg = nn.Sequential(
            nn.Linear(in_f, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        neigh: torch.Tensor,
        edge_h: torch.Tensor,
        cond_row: torch.Tensor,
        attn_tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (M, d)
        neigh: (M, K) long
        edge_h: (M, K, edge_dim)
        cond_row: (M, d)
        attn_tau: positive scalar tensor (,) for softmax temperature
        """
        h_n = self.norm_in(h)
        k = neigh.shape[1]
        hi = h_n.unsqueeze(1).expand(-1, k, -1)
        hj = h_n[neigh.long()]
        cond_e = cond_row.unsqueeze(1).expand(-1, k, -1)
        cat = torch.cat([hi, hj, edge_h, cond_e], dim=-1)
        msg = self.msg(cat)
        logits = self.score(msg).squeeze(-1) / attn_tau.clamp_min(1e-3)
        w = torch.softmax(logits, dim=-1).unsqueeze(-1)
        agg = (w * msg).sum(dim=1)
        return h + self.dropout(agg)


@torch.no_grad()
def _knn_indices_chunked(
    c_norm: torch.Tensor,
    k: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    kNN without O(M^2) distance matrix: process ``chunk_size`` query rows at a time.

    Peak memory O(chunk_size * M) instead of O(M^2) for full ``torch.cdist``.
    """
    m = c_norm.shape[0]
    kk = min(int(k), m - 1)
    if kk < 1:
        raise ValueError("knn k must be >= 1 when M > 1")
    device = c_norm.device
    dtype = c_norm.dtype
    neigh = torch.empty((m, kk), device=device, dtype=torch.long)
    cs = max(1, int(chunk_size))
    for start in range(0, m, cs):
        end = min(start + cs, m)
        chunk = c_norm[start:end]
        d = torch.cdist(chunk, c_norm)
        _, idx = d.topk(kk + 1, largest=False, dim=-1)
        neigh[start:end] = idx[:, 1:].to(torch.long)
        del d, idx, chunk
    return neigh


class GeoRelationalModule(nn.Module):
    """
    kNN graph on superpoint centroids; message passing conditioned on ``[SEG]`` text features.

    Returns ``sp_feats + gamma * tanh(Linear(LayerNorm(h_L)))`` with learnable ``gamma`` and
    learnable softmax temperature over neighbors.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        knn_k: int = 16,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        knn_chunk_size: int = 512,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.knn_chunk_size = max(1, int(knn_chunk_size))
        self.use_checkpoint = bool(use_checkpoint)
        self.num_layers = num_layers
        self.node_in = nn.Linear(in_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_in = nn.Linear(cond_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_GeoRelLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)
        # Positive attention temperature: tau = softplus(log) + tau_min
        self.log_attn_temp = nn.Parameter(torch.zeros(1))
        self.register_buffer("tau_min", torch.tensor(0.07))
        self.gamma = nn.Parameter(torch.tensor(0.02))
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def _attn_tau(self) -> torch.Tensor:
        return F.softplus(self.log_attn_temp) + self.tau_min.to(
            device=self.log_attn_temp.device, dtype=self.log_attn_temp.dtype
        )

    def forward(
        self,
        sp_feats: torch.Tensor,
        coords_float: torch.Tensor,
        superpoints: torch.Tensor,
        batch_offsets: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        sp_feats: (total_sp, in_dim)
        coords_float: (total_points, 3) same point order as superpoints indexing
        superpoints: (total_points,) long -> superpoint id
        batch_offsets: (B+1,) cumulative superpoint counts
        cond: (B, cond_dim) one row per scene (matches mask decoder batch when n_answers==1)
        """
        device = sp_feats.device
        dtype = sp_feats.dtype
        coords_float = coords_float.to(device=device, dtype=dtype)
        superpoints = superpoints.to(device=device, dtype=torch.long)

        with torch.no_grad():
            centroids = scatter_mean(coords_float, superpoints, dim=0)

        cond_h = self.cond_in(cond.to(dtype=dtype))
        tau = self._attn_tau()

        delta_acc = torch.zeros_like(sp_feats)
        bsz = int(batch_offsets.numel()) - 1

        for bi in range(bsz):
            s = int(batch_offsets[bi].item())
            e = int(batch_offsets[bi + 1].item())
            m = e - s
            if m <= 1:
                continue

            with torch.no_grad():
                c = centroids[s:e]
                c = c - c.mean(dim=0, keepdim=True)
                scale = c.std(dim=0, unbiased=False).clamp_min(1e-4)
                c_norm = (c / scale).to(dtype=dtype)
                neigh = _knn_indices_chunked(c_norm, self.knn_k, self.knn_chunk_size)
                delta = c_norm[neigh] - c_norm.unsqueeze(1)
                dist_e = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                edge_in = torch.cat([delta, torch.log1p(dist_e)], dim=-1).to(dtype=dtype)

            edge_h = self.edge_mlp(edge_in)

            h = self.node_in(sp_feats[s:e])
            cond_row = cond_h[bi : bi + 1].expand(m, -1)
            for layer in self.layers:
                if self.use_checkpoint and self.training:
                    h = checkpoint(
                        layer,
                        h,
                        neigh,
                        edge_h,
                        cond_row,
                        tau,
                        use_reentrant=False,
                    )
                else:
                    h = layer(h, neigh, edge_h, cond_row, tau)

            h_out = self.final_norm(h)
            raw = self.out(h_out)
            delta_acc[s:e] = self.gamma * torch.tanh(raw)

        return sp_feats + delta_acc
