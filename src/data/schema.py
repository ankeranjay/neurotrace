"""Core schemas — single source of truth for the whole project."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Concept:
    concept_id: str
    name: str
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class Question:
    question_id: str
    subject: str
    topic: str
    concepts: List[str]
    difficulty: int                       # 1..3
    question_type: str = "multiple_choice"
    correct_answer: str = ""
    prerequisite_concepts: List[str] = field(default_factory=list)


@dataclass
class Interaction:
    interaction_id: str
    student_id: str
    question_id: str
    concept_ids: List[str]
    is_correct: bool
    response_time_seconds: Optional[float] = None
    solution_steps: Optional[List[str]] = None
    timestamp: Optional[datetime] = None


@dataclass
class DiagnosisLabel:
    question_id: str
    error_category: str
    evidence: str
    annotator_confidence: float = 0.0


def to_json(obj: Any) -> Dict[str, Any]:
    d = asdict(obj)
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d