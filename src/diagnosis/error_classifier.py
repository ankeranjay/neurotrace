"""Hybrid error diagnosis (Contribution 2).

Rules first; ML classifier as fallback. If evidence insufficient → UNK.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional

# Error taxonomy (blueprint Section 4.5)
ERROR_TAXONOMY = [
    "E01_conceptual", "E02_algebraic_op", "E03_substitution",
    "E04_arithmetic", "E05_interpretation", "E06_method_selection",
    "E07_procedural", "UNK",
]

RULE_PATTERNS = {
    "E04_arithmetic": [r"\d+\s*[\+\-\*/]\s*\d+\s*=", r"\d+\.\d+"],
    "E02_algebraic_op": [r"x\s*=", r"\(.*\)\s*=", r"2x|x\^2"],
    "E03_substitution": [r"substitus", r"masukk", r"ganti"],
    "E05_interpretation": [r"diketahui", r"ditanya", r"kesimpulan"],
    "E07_procedural": [r"langkah", r"step", r"tahap"],
}


class ErrorClassifier:
    def __init__(self, min_evidence: int = 1):
        self.min_evidence = min_evidence

    def _rule_match(self, steps: List[str]) -> List[str]:
        hits = []
        text = " ".join(steps).lower()
        for cat, patterns in RULE_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                hits.append(cat)
        return hits

    def classify(
        self,
        question_concepts: List[str],
        solution_steps: Optional[List[str]],
        mastery_for_concepts: Optional[Dict[str, float]] = None,
    ) -> Dict:
        if not solution_steps:
            return {"category": "UNK", "confidence": 0.0,
                    "reason": "no_solution_steps"}

        rule_hits = self._rule_match(solution_steps)
        evidence_count = len(solution_steps)

        if evidence_count < self.min_evidence:
            return {"category": "UNK", "confidence": 0.1,
                    "reason": "insufficient_evidence"}

        if rule_hits:
            # Low mastery on prerequisites → likely conceptual
            if mastery_for_concepts:
                avg_m = sum(mastery_for_concepts.values()) / max(len(mastery_for_concepts), 1)
                if avg_m < 0.3:
                    return {"category": "E01_conceptual",
                            "confidence": 0.7, "reason": "low_prereq_mastery"}
            return {"category": rule_hits[0], "confidence": 0.6,
                    "reason": "rule_match"}

        # Fallback heuristic
        if mastery_for_concepts:
            avg = sum(mastery_for_concepts.values()) / len(mastery_for_concepts)
            if avg < 0.35:
                return {"category": "E01_conceptual", "confidence": 0.5,
                        "reason": "low_mastery"}
        return {"category": "UNK", "confidence": 0.2, "reason": "no_rule_hit"}