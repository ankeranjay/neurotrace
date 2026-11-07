"""Deep Knowledge Tracing — kembali ke konfigurasi iter 4 (AUC 0.6379).

Perubahan dari iter 5:
  - hapus init_h / init_c dari konsep pertama (sumber noise)
  - response_emb 2-way (bukan 3-way), sesuai iter 4
"""
import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, num_concepts: int, hidden: int = 48, dropout: float = 0.4):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_emb = nn.Embedding(num_concepts, hidden)
        self.response_emb = nn.Embedding(2, hidden)
        self.lstm = nn.LSTM(hidden * 2, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, c_seq: torch.Tensor, r_seq: torch.Tensor) -> torch.Tensor:
        # Shift response: predict r_t from r_{t-1}
        r_prev = torch.zeros_like(r_seq)
        r_prev[:, 1:] = r_seq[:, :-1]

        x = torch.cat([
            self.concept_emb(c_seq),
            self.response_emb(r_prev.long())
        ], dim=-1)

        out, _ = self.lstm(x)                     # h0, c0 = zeros (default)
        logits = self.head(self.dropout(out)).squeeze(-1)
        return logits