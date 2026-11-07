from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="NeuroTrace API")


class InteractionIn(BaseModel):
    concepts: List[int]
    responses: List[int]


class DiagnoseIn(BaseModel):
    solution_steps: List[str]
    concepts: List[str]
    evidence_count: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(data: InteractionIn):
    # TODO: load model, run inference
    return {"probability": 0.5, "uncertainty": 0.25,
            "note": "stub — wire model from scripts/run_pipeline"}


@app.post("/diagnose")
def diagnose_endpoint(data: DiagnoseIn):
    from src.diagnosis.evidence_validator import EvidenceValidator
    if data.evidence_count < 2:
        return {"accepted": False, "category": "UNK",
                "reason": "insufficient_evidence"}
    return {"accepted": True, "category": "E04_arithmetic"}