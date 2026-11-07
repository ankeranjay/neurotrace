"""Ablation study configs (blueprint Section 8)."""
import torch
from ..models.neurotrace import UAGKT
from ..models.dkt import DKT


def build_model(config: str, num_concepts: int, hidden: int, gnn_layers: int):
    """
    A: Temporal ML only              → DKT
    B: ML + KG                       → GKT (unweighted graph)
    C: ML + Uncertainty              → DKT + dropout head
    D: ML + KG + Uncertainty         → UAG-KT (proposed)
    E: ML + KG + LLM                 → KG + LLM layer (post-hoc, use GKT)
    F: Full NeuroTrace               → UAG-KT + LLM layer
    """
    if config == "A":
        return DKT(num_concepts, hidden)
    if config == "B":
        from ..models.gkt import GKT
        return GKT(num_concepts, hidden, gnn_layers)
    if config in ("C", "D", "E", "F"):
        return UAGKT(num_concepts, hidden, gnn_layers)
    raise ValueError(config)