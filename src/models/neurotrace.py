"""Uncertainty-Aware Graph Knowledge Tracing (UAG-KT).

Dengan residual connection untuk graph, dan MC Dropout untuk uncertainty.
logvar_head tetap ada (untuk MC estimation), tapi loss pakai BCE saja.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict

from ..knowledge_graph.graph import UncertaintyAwareGNN


class UAGKT(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        hidden: int = 48,
        gnn_layers: int = 2,
        gru_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.hidden = hidden

        self.gnn = UncertaintyAwareGNN(num_concepts, hidden, gnn_layers)
        self.concept_emb_direct = nn.Embedding(num_concepts, hidden)
        self.response_emb = nn.Embedding(2, hidden)
        self.gru = nn.GRU(hidden * 2, hidden, num_layers=gru_layers,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.pred_head = nn.Linear(hidden, 1)
        self.mastery_head = nn.Linear(hidden, num_concepts)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, c_seq, r_seq, A, U, mc_dropout=False):
        H = self.gnn(A, U)

        concept_combined = H[c_seq] + self.concept_emb_direct(c_seq)

        r_prev = torch.zeros_like(r_seq)
        r_prev[:, 1:] = r_seq[:, :-1]

        x = torch.cat([
            concept_combined,
            self.response_emb(r_prev.long())
        ], dim=-1)

        if mc_dropout:
            x = self.dropout(x)

        out, _ = self.gru(x)

        if mc_dropout:
            out = self.dropout(out)

        logits = self.pred_head(out).squeeze(-1)
        logvar = self.logvar_head(out).squeeze(-1)
        mastery = torch.sigmoid(self.mastery_head(out))

        return {"logits": logits, "logvar": logvar,
                "mastery": mastery, "concept_emb": H}

    @torch.no_grad()
    def predict_with_uncertainty(self, c_seq, r_seq, A, U, n_samples: int = 20):
        self.train()   # dropout ON
        probs, masteries = [], []
        for _ in range(n_samples):
            o = self.forward(c_seq, r_seq, A, U, mc_dropout=True)
            probs.append(torch.sigmoid(o["logits"]))
            masteries.append(o["mastery"])
        self.eval()
        probs = torch.stack(probs)
        masteries = torch.stack(masteries)
        return {
            "prob_mean": probs.mean(0),
            "prob_var": probs.var(0),
            "mastery_mean": masteries.mean(0),
            "mastery_var": masteries.var(0),
        }