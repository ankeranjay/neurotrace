"""BKT per-skill — standar literatur Knowledge Tracing.

Setiap skill punya model BKT terpisah dengan parameter optimal sendiri.
Setup ini sama dengan:
  - Corbett & Anderson 1994 (BKT asli)
  - Piech et al. 2015 (DKT paper, BKT baseline)
  - Semua paper KT setelahnya

Keuntungan vs BKT global:
  - 4 parameter per skill (137 skill × 4 = 548 parameter)
  - Tidak "cheat" dengan melacak ability umum siswa
  - Tidak stuck di batas bounds MLE
  - Hasil comparable ke literatur (AUC ~0.70–0.78)
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize


class _BKTModel:
    """BKT single-skill, numerically stable (sama dengan bkt.py)."""

    def __init__(self, p_L0=0.3, p_T=0.1, p_S=0.1, p_G=0.2):
        self.p = np.array([p_L0, p_T, p_S, p_G], dtype=np.float64)

    def _forward(self, seq: List[int], p=None) -> List[float]:
        p_L0, p_T, p_S, p_G = p if p is not None else self.p
        p_L0 = np.clip(p_L0, 1e-4, 1 - 1e-4)
        p_T = np.clip(p_T, 1e-4, 1 - 1e-4)
        p_S = np.clip(p_S, 1e-4, 1 - 1e-4)
        p_G = np.clip(p_G, 1e-4, 1 - 1e-4)

        p_L = p_L0
        out = []
        for r in seq:
            p_c = p_L * (1 - p_S) + (1 - p_L) * p_G
            p_c = float(np.clip(p_c, 1e-6, 1 - 1e-6))
            out.append(p_c)

            if r == 1:
                num = p_L * (1 - p_S)
                den = p_c
            else:
                num = p_L * p_S
                den = 1 - p_c
            den = max(den, 1e-6)
            p_post = num / den
            p_post = float(np.clip(p_post, 1e-6, 1 - 1e-6))

            p_L = p_post + (1 - p_post) * p_T
            p_L = float(np.clip(p_L, 1e-6, 1 - 1e-6))
        return out

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


class BKTPerSkill:
    """Wrapper: satu BKT model per skill."""

    def __init__(self, num_skills: int, min_sequences: int = 10):
        self.num_skills = num_skills
        self.min_sequences = min_sequences
        self.models: Dict[int, _BKTModel] = {}
        self.global_fallback: float = 0.5

    def fit(self, sequences_with_skill: List[List[Tuple[int, int]]]):
        """
        sequences_with_skill: list per siswa, tiap elemen (skill_id, is_correct).
        """
        per_skill: Dict[int, List[List[int]]] = {}
        all_correct, all_total = 0, 0

        for seq in sequences_with_skill:
            skill_seq: Dict[int, List[int]] = {}
            for skill, correct in seq:
                skill_seq.setdefault(skill, []).append(int(correct))
            for skill, resp in skill_seq.items():
                per_skill.setdefault(skill, []).append(resp)
                all_correct += sum(resp)
                all_total += len(resp)

        fitted = 0
        for skill, skill_seqs in per_skill.items():
            if len(skill_seqs) < self.min_sequences:
                continue
            self.models[skill] = _BKTModel().fit(skill_seqs)
            fitted += 1

        if all_total > 0:
            self.global_fallback = all_correct / all_total

        print(f"  Fitted {fitted}/{len(per_skill)} skill models "
              f"(min_seq={self.min_sequences})")
        print(f"  Global fallback (base rate): {self.global_fallback:.3f}")
        return self

    def predict_sequence(self, seq: List[Tuple[int, int]]) -> List[float]:
        """
        Predict per-step, dengan temporal tracking p_L per skill.
        Siswa punya multiple skill → track p_L per skill independently.
        """
        p_L_state: Dict[int, float] = {}
        preds = []

        for skill, correct in seq:
            if skill not in self.models:
                preds.append(self.global_fallback)
                continue

            model = self.models[skill]
            p_L0, p_T, p_S, p_G = model.p

            if skill not in p_L_state:
                p_L_state[skill] = float(p_L0)
            p_L = p_L_state[skill]

            p_c = p_L * (1 - p_S) + (1 - p_L) * p_G
            p_c = float(np.clip(p_c, 1e-6, 1 - 1e-6))
            preds.append(p_c)

            if correct == 1:
                num = p_L * (1 - p_S)
                den = p_c
            else:
                num = p_L * p_S
                den = 1 - p_c
            den = max(den, 1e-6)
            p_post = num / den
            p_post = float(np.clip(p_post, 1e-6, 1 - 1e-6))

            p_L_new = p_post + (1 - p_post) * p_T
            p_L_new = float(np.clip(p_L_new, 1e-6, 1 - 1e-6))
            p_L_state[skill] = p_L_new

        return preds

    def param_summary(self) -> Dict[str, float]:
        """Ringkasan parameter untuk sanity check."""
        if not self.models:
            return {}
        params = np.array([m.p for m in self.models.values()])
        return {
            "num_models": len(self.models),
            "p_L0_mean": float(params[:, 0].mean()),
            "p_T_mean": float(params[:, 1].mean()),
            "p_S_mean": float(params[:, 2].mean()),
            "p_G_mean": float(params[:, 3].mean()),
            "p_S_pG_mean": float((params[:, 2] + params[:, 3]).mean()),
        }