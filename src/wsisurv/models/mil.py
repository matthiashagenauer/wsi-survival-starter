"""Attention-based MIL (Ilse et al. 2018, gated variant) for slide/patient-level prediction.

    tiles (N, D) --Linear+GELU--> (N, H) --gated attention--> weights a_i (softmax over tiles)
                                         --sum_i a_i h_i--> patient vector (H) --Linear--> outputs

outputs = 1 risk score (Cox loss) or K hazard logits (discrete-time NLL loss).
`pooling="mean"` replaces attention by the plain average — a baseline worth keeping.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ABMIL(nn.Module):
    def __init__(self, input_dim: int, n_outputs: int = 1, hidden_dim: int = 256,
                 dropout: float = 0.25, pooling: str = "attention"):
        super().__init__()
        if pooling not in {"attention", "mean"}:
            raise ValueError(f"pooling must be 'attention' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.attn_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, n_outputs))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """x: (B, N, D) -> (B, n_outputs) [, attention (B, N)]"""
        h = self.embed(x)
        if self.pooling == "mean":
            a = torch.full(h.shape[:2], 1.0 / h.shape[1], device=h.device, dtype=h.dtype)
        else:
            a = torch.softmax(self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1), dim=1)
        z = torch.einsum("bn,bnh->bh", a, h)
        out = self.head(z)
        return (out, a) if return_attention else out
