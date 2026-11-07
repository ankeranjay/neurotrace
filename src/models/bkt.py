"""Bayesian Knowledge Tracing — stabil secara numerik."""
from __future__ import annotations
from typing import List
import numpy as np
from scipy.optimize import minimize


class BKT:
    def __init__(self, p_L0=0.3, p_T=0.1, p_S=0.1, p_G=0.2):
        self.p = np.array([p_L0, p_T, p_S, p_G], dtype=np.float64)

    def _forward(self, seq: List[int], p=None) -> List[float]:
        p_L0, p_T, p_S, p_G = p if p is not None else self.p
        # Clamp parameter ke range aman
        p_L0 = np.clip(p_L0, 1e-4, 1 - 1e-4)
        p_T = np.clip(p_T, 1e-4, 1 - 1e-4)
        p_S = np.clip(p_S, 1e-4, 1 - 1e-4)
        p_G = np.clip(p_G, 1e-4, 1 - 1e-4)

        p_L = p_L0
        out = []
        for r in seq:
            p_c = p_L * (1 - p_S) + (1 - p_L) * p_G
            # Clamp p_c agar tidak 0 atau 1
            p_c = float(np.clip(p_c, 1e-6, 1 - 1e-6))
            out.append(p_c)

            if r == 1:
                num = p_L * (1 - p_S)
                den = p_c
            else:
                num = p_L * p_S
                den = 1 - p_c

            # Clamp denominator agar tidak 0
            den = max(den, 1e-6)
            p_post = num / den
            p_post = float(np.clip(p_post, 1e-6, 1 - 1e-6))

            p_L = p_post + (1 - p_post) * p_T
            p_L = float(np.clip(p_L, 1e-6, 1 - 1e-6))

        return out

    def predict(self, seq):
        return self._forward(seq)

    def fit(self, sequences: List[List[int]]):
        def nll(p):
            if not np.all((p > 1e-4) & (p < 1 - 1e-4)):
                return 1e12
            tot = 0.0
            for s in sequences:
                try:
                    preds = self._forward(s, p=p)
                except Exception:
                    return 1e12
                for r, pr in zip(s, preds):
                    pr = float(np.clip(pr, 1e-12, 1 - 1e-12))
                    tot -= np.log(pr if r else 1 - pr)
            if np.isnan(tot) or np.isinf(tot):
                return 1e12
            return tot

        res = minimize(nll, self.p, method="L-BFGS-B",
                       bounds=[(0.01, 0.99)] * 4)
        self.p = res.x
        return self