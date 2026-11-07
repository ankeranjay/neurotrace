"""Uncertainty-aware next-question selector (Contribution 3)."""
from __future__ import annotations
import numpy as np
from typing import Dict, List
from ..knowledge_graph.graph import KnowledgeGraph


def select_next(
    candidates: List[Dict],
    mastery: Dict[str, float],
    mastery_var: Dict[str, float],
    kg: KnowledgeGraph,
    target_mastery: float = 0.75,
) -> Dict:
    """
    Score each candidate. Higher = better.
    Balances: ZPD gap, prerequisite readiness, uncertainty reduction.
    """
    prereq_of = {c: kg.ids[i] for c, i in kg.idx.items()}  # dummy safe
    scores = []

    for cand in candidates:
        concepts = cand["concepts"]
        c = concepts[0]
        m = mastery.get(c, 0.5)
        var = mastery_var.get(c, 0.25)

        # 1) Zone of Proximal Development: prefer ~0.5 mastery
        zpd_gap = -abs(m - 0.5)

        # 2) Prerequisite readiness
        prereqs = kg.ids  # ignore if you don't have edge index backwards
        prereq_ready = 1.0
        for p in kg.ids:
            if kg.A[kg.idx[p], kg.idx[c]] > 0:
                prereq_ready *= max(mastery.get(p, 0.3), 0.2)

        # 3) Uncertainty reduction potential
        info_gain = var * (1.0 - m)

        # 4) Difficulty alignment (still simple)
        diff_pen = -abs(cand.get("difficulty", 2) - 2)

        score = 1.0 * zpd_gap + 0.8 * prereq_ready + 0.6 * info_gain + 0.1 * diff_pen
        scores.append((score, cand))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1] if scores else {}