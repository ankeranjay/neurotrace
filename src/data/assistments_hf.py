"""Loader Atomi/ASSISTments2009 dari HuggingFace.

CATATAN PENTING soal kolom (dataset HF ini punya nama menyesatkan):
  - skill_ids      : list[int]    ID skill (BENAR — dipakai sebagai concept_ids)
  - skill_names    : list[str]    nama skill
  - grades         : list[str]    "0"/"1" → label benar/salah (BENAR — dipakai sebagai label)
  - answer_types   : list[str]    isinya nama skill (MISLEADING — JANGAN dipakai label!)
  - attempt_counts : list[int]    jumlah percobaan
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .schema import Interaction


def load_assistments_hf(
    n_students: int | None = None,
    max_len: int = 100,
    min_interactions: int = 10,
    verbose: bool = True,
) -> Tuple[List[List[Interaction]], int, np.ndarray]:
    """Load ASSISTments2009 dari HF.

    Label diambil dari kolom `grades` ("0"/"1").
    """
    from datasets import load_dataset

    if verbose:
        print(f"Loading ASSISTments2009 dari HuggingFace "
              f"(n_students={n_students or 'all'}, max_len={max_len})...")

    ds = load_dataset("Atomi/ASSISTments2009", split="train")

    sequences: List[List[Interaction]] = []
    all_skills = set()

    for row in ds:
        if n_students is not None and len(sequences) >= n_students:
            break

        skill_ids = row["skill_ids"]
        grades = row["grades"]

        if len(skill_ids) < min_interactions:
            continue
        if len(skill_ids) != len(grades):
            continue   # data rusak, skip

        answers = [1 if str(g) == "1" else 0 for g in grades]

        user_id = row["user_id"]
        n_take = min(len(skill_ids), max_len)
        seq: List[Interaction] = []
        for j in range(n_take):
            sid = skill_ids[j]
            ans = answers[j]
            seq.append(Interaction(
                interaction_id=f"{user_id}-{j}",
                student_id=f"ANON-{user_id}",
                question_id=f"q-{user_id}-{j}",
                concept_ids=[str(sid)],
                is_correct=bool(ans),
                response_time_seconds=None,
            ))
            all_skills.add(str(sid))

        if seq:
            sequences.append(seq)

    # Remap skill IDs ke 0..K-1
    sorted_skills = sorted(
        all_skills,
        key=lambda x: int(x) if str(x).isdigit() else 0,
    )
    skill_to_idx = {s: i for i, s in enumerate(sorted_skills)}
    K = len(sorted_skills)

    for seq in sequences:
        for it in seq:
            it.concept_ids = [str(skill_to_idx[it.concept_ids[0]])]

    if verbose:
        lens = [len(s) for s in sequences]
        total = sum(lens)
        correct_frac = (
            sum(it.is_correct for s in sequences for it in s) / max(total, 1)
        )
        print(f"  Students: {len(sequences):,} | K={K} skills | "
              f"interactions={total:,} | mean_len={np.mean(lens):.1f} | "
              f"correct_rate={correct_frac:.3f}")
        if correct_frac < 0.4 or correct_frac > 0.85:
            print(f"  ⚠️ WARNING: correct_rate={correct_frac:.2f} tidak wajar. "
                  f"Cek kembali field grades.")
        else:
            print(f"  ✅ Label benar (correct_rate ~0.5–0.8)")

    A_np = np.eye(K, dtype=np.float32)
    return sequences, K, A_np


def build_cooccurrence_graph(
    sequences: List[List[Interaction]],
    num_concepts: int,
    threshold: float = 0.05,
) -> np.ndarray:
    """Bangun adjacency dari co-occurrence skill berurutan."""
    cooc = np.zeros((num_concepts, num_concepts), dtype=np.float32)
    for seq in sequences:
        concepts = [int(s.concept_ids[0]) for s in seq]
        for i in range(len(concepts) - 1):
            a, b = concepts[i], concepts[i + 1]
            if a != b and 0 <= a < num_concepts and 0 <= b < num_concepts:
                cooc[a, b] += 1.0

    row_sums = cooc.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cooc_prob = cooc / row_sums

    A = (cooc_prob > threshold).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A