"""Synthetic dataset dengan signal yang kuat untuk Knowledge Tracing.

Prinsip desain:
  - Setiap siswa punya latent mastery per konsep.
  - Konsep diulang dalam blok pendek (5 langkah) supaya model bisa
    belajar dari respons sebelumnya pada konsep yang sama.
  - Mastery naik setiap kali (benar atau salah), lebih besar kalau benar.
  - Noise slip/guess kecil supaya pola terlihat.
"""
from __future__ import annotations
import uuid
import numpy as np
from typing import List, Tuple, Dict

from .schema import Interaction


SCENARIOS = [
    "always_correct",
    "consistent_learner",
    "inconsistent",
    "topic_split",
    "cold_start",
]


def generate(
    n_students: int = 500,
    n_concepts: int = 8,
    n_steps: int = 40,
    scenario: str = "consistent_learner",
    slip: float = 0.08,
    guess: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, List[List[Interaction]], np.ndarray]:
    rng = np.random.default_rng(seed)

    # Prerequisite chain
    A = np.zeros((n_concepts, n_concepts))
    for i in range(n_concepts - 1):
        A[i, i + 1] = 1.0

    # Mastery awal: bimodal (beberapa siswa kuat, beberapa lemah)
    mastery = rng.beta(3, 3, size=(n_students, n_concepts))

    if scenario == "always_correct":
        mastery = np.ones_like(mastery)
    elif scenario == "topic_split":
        half = n_concepts // 2
        mastery[:, :half] = rng.beta(6, 2, size=(n_students, half))
        mastery[:, half:] = rng.beta(2, 6, size=(n_students, n_concepts - half))
    elif scenario == "cold_start":
        n_steps = 5

    sequences: List[List[Interaction]] = []
    for sid in range(n_students):
        student_id = f"ANON-{sid:04d}"

        # Setiap siswa punya SATU konsep fokus yang di-drill
        # (realistis: siswa latihan topik yang sama berulang)
        focus = int(rng.integers(0, n_concepts))

        seq: List[Interaction] = []
        for t in range(n_steps):
            # 70% konsep fokus, 30% konsep lain (eksplorasi)
            if rng.random() < 0.7:
                c = focus
            else:
                c = int(rng.integers(0, n_concepts))

            p_mastery = float(mastery[sid, c])

            if scenario == "inconsistent":
                # Model tidak bisa menebak → baseline harus ~0.5
                correct = bool(rng.random() < 0.5)
            else:
                # Noisy response model (standar BKT)
                if rng.random() < p_mastery:
                    correct = rng.random() > slip
                else:
                    correct = rng.random() < guess

            # Learning update: mastery naik setiap langkah
            # (benar → naik lebih besar; salah → naik sedikit)
            delta = 0.04 if correct else 0.015
            mastery[sid, c] = min(1.0, p_mastery + delta)

            seq.append(Interaction(
                interaction_id=f"INT-{uuid.uuid4().hex[:10]}",
                student_id=student_id,
                question_id=f"Q-{c}-{t}",
                concept_ids=[str(c)],
                is_correct=bool(correct),
                response_time_seconds=float(rng.uniform(30, 180)),
            ))

        sequences.append(seq)

    return mastery, sequences, A


def to_tensors(sequences: List[List[Interaction]],
               max_concepts: int,
               max_len: int = 50) -> Dict[str, np.ndarray]:
    n = len(sequences)
    c_out = np.zeros((n, max_len), dtype=np.int64)
    r_out = np.zeros((n, max_len), dtype=np.float32)
    m_out = np.zeros((n, max_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, it in enumerate(seq[:max_len]):
            c_out[i, j] = int(it.concept_ids[0]) % max_concepts
            r_out[i, j] = float(it.is_correct)
            m_out[i, j] = 1.0
    return {"concepts": c_out, "responses": r_out, "mask": m_out}