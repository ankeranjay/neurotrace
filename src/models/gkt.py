"""Graph Knowledge Tracing dengan residual connection.

Ide: H[c_seq] (graph) + concept_emb_direct(c_seq) (langsung).
Model bisa "skip" graph kalau tidak membantu, tapi tetap memakai
informasi graph kalau berguna. Ini seperti ResNet.
"""
import torch
import torch.nn as nn
from ..knowledge_graph.graph import UncertaintyAwareGNN


class GKT(nn.Module):
    def __init__(self, num_concepts: int, hidden: int = 48,
                 gnn_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.gnn = UncertaintyAwareGNN(num_concepts, hidden, gnn_layers)
        # Residual: embedding langsung (bisa belajar tanpa graph)
        self.concept_emb_direct = nn.Embedding(num_concepts, hidden)
        self.response_emb = nn.Embedding(2, hidden)
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, c_seq, r_seq, A, U, mc_dropout=False):
        H = self.gnn(A, U)                              # (K, h)

        # Residual connection: graph + direct embedding
        concept_combined = H[c_seq] + self.concept_emb_direct(c_seq)

        r_prev = torch.zeros_like(r_seq)
        r_prev[:, 1:] = r_seq[:, :-1]

        x = torch.cat([
            concept_combined,
            self.response_emb(r_prev.long())
        ], dim=-1)

        out, _ = self.gru(x)
        logits = self.head(self.dropout(out)).squeeze(-1)
        return {"logits": logits, "concept_emb": H}