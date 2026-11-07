"""Uncertainty-aware Knowledge Graph (Contribution 1)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class KnowledgeGraph:
    """Directed concept graph with per-edge confidence weights."""

    def __init__(self, concept_ids: List[str]):
        self.ids = list(concept_ids)
        self.idx = {c: i for i, c in enumerate(self.ids)}
        k = len(self.ids)
        self.A = np.zeros((k, k), dtype=np.float32)      # adjacency
        self.U = np.ones((k, k), dtype=np.float32)       # uncertainty [0,1], 1 = unknown

    def add_edge(self, src: str, dst: str, weight: float, confidence: float = 1.0):
        if src not in self.idx or dst not in self.idx:
            raise KeyError(f"unknown concept: {src}->{dst}")
        i, j = self.idx[src], self.idx[dst]
        self.A[i, j] = weight
        self.U[i, j] = 1.0 - confidence

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.A), torch.from_numpy(self.U)

    @property
    def num_concepts(self) -> int:
        return len(self.ids)


# ---------- GNN with uncertainty ----------

class UncertaintyGCN(nn.Module):
    """Message passing weighted by edge confidence."""
    def __init__(self, hidden: int):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.bn = nn.LayerNorm(hidden)

    def forward(self, H: torch.Tensor, A: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # Confidence-weighted adjacency
        A_conf = A * (1.0 - U)
        deg = A_conf.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A_conf / deg                       # (K, K)
        msg = A_norm @ H                            # (K, d)
        out = self.bn(self.W(msg) + H)              # residual
        return F.relu(out)


class UncertaintyAwareGNN(nn.Module):
    def __init__(self, num_concepts: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(num_concepts, hidden)
        self.layers = nn.ModuleList([UncertaintyGCN(hidden) for _ in range(layers)])

    def forward(self, A: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        H = self.embed.weight
        for layer in self.layers:
            H = layer(H, A, U)
        return H                                    # (K, d)