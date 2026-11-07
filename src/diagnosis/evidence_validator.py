"""Ensures diagnosis is backed by minimum evidence (abstention)."""
from typing import Dict


class EvidenceValidator:
    def __init__(self, min_evidence: int = 2, min_confidence: float = 0.4):
        self.min_evidence = min_evidence
        self.min_conf = min_confidence

    def validate(self, diagnosis: Dict, evidence_count: int) -> Dict:
        if diagnosis["category"] == "UNK":
            return {"accepted": False, "reason": "model_unk"}
        if evidence_count < self.min_evidence:
            return {"accepted": False, "reason": "insufficient_evidence",
                    "downgraded_to": "UNK"}
        if diagnosis["confidence"] < self.min_conf:
            return {"accepted": False, "reason": "low_confidence",
                    "downgraded_to": "UNK"}
        return {"accepted": True, "reason": "ok"}