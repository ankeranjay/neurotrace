"""LLM reasoning layer — structured, evidence-grounded, with abstention."""
from __future__ import annotations
import json
from typing import Dict, Optional, Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


PROMPT_TEMPLATE = """You are an educational assistant for a math tutoring system.

STRICT RULES:
- Base every claim on EVIDENCE provided below.
- If EVIDENCE_COUNT < 2, respond exactly with: {"abstain": true, "reason": "insufficient_evidence"}
- Do NOT invent steps not present in EVIDENCE.
- Output valid JSON only.

EVIDENCE:
{evidence_json}

TASK:
Explain the learner's likely error and suggest ONE remedial concept.
Return JSON with keys: explanation (string), suggested_concept (string), confidence (0-1).
"""


class LLMReasoner:
    def __init__(self, client: LLMClient, min_evidence: int = 2):
        self.client = client
        self.min_evidence = min_evidence

    def explain(self, payload: Dict) -> Dict:
        evidence_count = payload.get("evidence_count", 0)
        if evidence_count < self.min_evidence:
            return {"abstain": True, "reason": "insufficient_evidence"}

        prompt = PROMPT_TEMPLATE.format(
            evidence_json=json.dumps(payload, indent=2)
        )
        try:
            raw = self.client.complete(prompt)
            data = json.loads(raw)
        except Exception as e:
            return {"abstain": True, "reason": f"parse_error:{e}"}

        # Validate structure
        if not isinstance(data, dict) or "explanation" not in data:
            return {"abstain": True, "reason": "invalid_structure"}
        return data