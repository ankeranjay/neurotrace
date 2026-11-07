"""Bayesian Knowledge Tracing — classical baseline."""
from __future__ import annotations
from typing import List
import numpy as np
from scipy.optimize import minimize


class BKT:
    def __init__(self, p_L0=0.3, p_T=0.1, p_S=0.1, p_G=0.2):
        self.p = np.array([p_L0, p_T, p_S, p_G], dtype=np.float64)

    def _forward(self, seq: List[int], p=None) -> List[float]:
        p_L0, p_T, p_S, p_G = p if p is not None else self.p
        p_L = p_L0
        out = []
        for r in seq:
            p_c = p_L * (1 - p_S) + (1 - p_L) * p_G
            out.append(p_c)
            if r == 1:
                p_post = (p_L * (1 - p_S)) / max(p_c, 1e-12)
            else:
                p_post = (p_L * p_S) / max(1 - p_c, 1e-12)
            p_L = p_post + (1 - p_post) * p_T
        return out

    def predict(self, seq): return self._forward(seq)

    def fit(self, sequences: List[List[int]]):
        def nll(p):
            if not np.all((p > 1e-4) & (p < 1 - 1e-4)):
                return 1e12
            tot = 0.0
            for s in sequences:
                preds = self._forward(s, p=p)
                for r, pr in zip(s, preds):
                    pr = np.clip(pr, 1e-12, 1 - 1e-12)
                    tot -= np.log(pr if r else 1 - pr)
            return tot
        res = minimize(nll, self.p, method="L-BFGS-B",
                       bounds=[(0.01, 0.99)] * 4)
        self.p = res.x
        return self