"""Utilities: Ensemble + calibration helpers."""
import torch
from typing import List


class DeepEnsemble:
    def __init__(self, models: List[torch.nn.Module]):
        self.models = models

    @torch.no_grad()
    def predict(self, c_seq, r_seq, A, U):
        self.models[0].eval()
        preds = []
        masteries = []
        for m in self.models:
            m.eval()
            o = m(c_seq, r_seq, A, U)
            preds.append(torch.sigmoid(o["logits"]))
            masteries.append(o["mastery"])
        preds = torch.stack(preds)
        masteries = torch.stack(masteries)
        return {
            "prob_mean": preds.mean(0),
            "prob_var": preds.var(0),
            "mastery_mean": masteries.mean(0),
            "mastery_var": masteries.var(0),
        }