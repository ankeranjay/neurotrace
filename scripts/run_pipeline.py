"""Pipeline end-to-end — synthetic / ASSISTments CSV / HuggingFace.

Usage:
  python -m scripts.run_pipeline
  python -m scripts.run_pipeline --data hf --n-students 5000 --max-len 100 --epochs 150
  python -m scripts.run_pipeline --data hf --bkt-mode per-skill
  python -m scripts.run_pipeline --data hf --bkt-mode global
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.synthetic import generate, to_tensors
from src.knowledge_graph.graph import KnowledgeGraph
from src.models.bkt import BKT
from src.models.dkt import DKT
from src.models.gkt import GKT
from src.models.neurotrace import UAGKT
from src.training.train import train_model, predict
from src.evaluation.metrics import binary_metrics, expected_calibration_error


def _build_skill_sequences(tensors, indices):
    """Convert tensors (concepts, responses, mask) → list per siswa
    berisi (skill_id, is_correct)."""
    concepts = tensors["concepts"][indices]
    responses = tensors["responses"][indices]
    mask = tensors["mask"][indices]
    out = []
    for i in range(len(concepts)):
        seq = []
        for j in range(concepts.shape[1]):
            if mask[i, j] > 0:
                seq.append((int(concepts[i, j]), int(responses[i, j])))
        if seq:
            out.append(seq)
    return out


def _eval_bkt_global(train_t, test_t):
    train_seq = [
        [int(x) for x in r]
        for r, m in zip(train_t["responses"].numpy(), train_t["mask"].numpy())
        if m.any()
    ]
    bkt = BKT().fit(train_seq)
    test_seq = [
        [int(x) for x in r]
        for r, m in zip(test_t["responses"].numpy(), test_t["mask"].numpy())
        if m.any()
    ]
    y_true, y_prob = [], []
    for seq in test_seq:
        y_true += seq
        y_prob += bkt.predict(seq)
    return y_true, y_prob, bkt


def _eval_bkt_per_skill(tensors, tr_idx, te_idx, K):
    from src.models.bkt_per_skill import BKTPerSkill
    train_skill = _build_skill_sequences(tensors, tr_idx)
    test_skill = _build_skill_sequences(tensors, te_idx)
    bkt = BKTPerSkill(K).fit(train_skill)
    summary = bkt.param_summary()
    print(f"  Param summary: {summary}")
    y_true, y_prob = [], []
    for seq in test_skill:
        preds = bkt.predict_sequence(seq)
        for (skill, correct), p in zip(seq, preds):
            y_true.append(correct)
            y_prob.append(p)
    return y_true, y_prob, bkt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "hf"], default="synthetic")
    parser.add_argument("--n-students", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--bkt-mode", choices=["global", "per-skill"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Default: per-skill untuk hf, global untuk synthetic
    if args.bkt_mode is None:
        args.bkt_mode = "per-skill" if args.data == "hf" else "global"

    H = args.hidden

    # ============================================================
    # 1. Load data
    # ============================================================
    if args.data == "hf":
        print("=== 1. Load ASSISTments (HF) ===")
        from src.data.assistments_hf import (
            load_assistments_hf, build_cooccurrence_graph,
        )
        sequences, K, _ = load_assistments_hf(
            n_students=args.n_students, max_len=args.max_len,
        )
        A_np = build_cooccurrence_graph(sequences, K, threshold=0.05)
        MAX_LEN = min(args.max_len, max(len(s) for s in sequences))
        print(f"Graph: K={K} | edges={int(A_np.sum())}")
        EPOCHS = args.epochs or 100
    else:
        print("=== 1. Generate synthetic data ===")
        K = 8
        MAX_LEN = 40
        mastery, sequences, A_np = generate(
            n_students=500, n_concepts=K, n_steps=MAX_LEN
        )
        EPOCHS = args.epochs or 80

    tensors = to_tensors(sequences, max_concepts=K, max_len=MAX_LEN)
    print(f"Sequences: {len(sequences)} | K={K} | max_len={MAX_LEN} | "
          f"epochs={EPOCHS} | batch={args.batch_size} | bkt={args.bkt_mode}")

    n = len(sequences)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    def slice_t(d, ids):
        return {k: torch.from_numpy(v[ids]) for k, v in d.items()}

    train_t = slice_t(tensors, tr_idx)
    test_t = slice_t(tensors, te_idx)

    # ============================================================
    # 2. Graph
    # ============================================================
    kg = KnowledgeGraph([str(i) for i in range(K)])
    for i in range(K):
        for j in range(K):
            if A_np[i, j] > 0:
                kg.add_edge(str(i), str(j),
                            weight=float(A_np[i, j]), confidence=1.0)
    A_t, U_t = kg.to_tensors()
    U_t = torch.zeros_like(U_t)

    results = {}

    # ============================================================
    # 3. BKT
    # ============================================================
    print(f"\n=== 2. BKT ({args.bkt_mode}) ===")
    if args.bkt_mode == "per-skill":
        y_true, y_prob, bkt = _eval_bkt_per_skill(tensors, tr_idx, te_idx, K)
    else:
        y_true, y_prob, bkt = _eval_bkt_global(train_t, test_t)

    results["BKT"] = binary_metrics(y_true, y_prob)
    results["BKT"]["ece"] = expected_calibration_error(
        np.array(y_true, dtype=bool), np.array(y_prob)
    )
    print(f"BKT: {results['BKT']}")

    # ============================================================
    # 4. DKT
    # ============================================================
    print("\n=== 3. DKT ===")
    dkt = DKT(K, hidden=H, dropout=0.4)
    train_model(dkt, train_t, A_t, U_t, epochs=EPOCHS,
                batch_size=args.batch_size, weight_decay=5e-3,
                is_graph_model=False)
    p_tr, y_tr, m_tr = predict(dkt, train_t, A_t, U_t, is_graph_model=False)
    auc_tr = binary_metrics(y_tr, p_tr, m_tr)["auc"]
    p, y, m = predict(dkt, test_t, A_t, U_t, is_graph_model=False)
    results["DKT"] = binary_metrics(y, p, m)
    results["DKT"]["ece"] = expected_calibration_error(
        y[m.astype(bool)], p[m.astype(bool)]
    )
    print(f"DKT train AUC: {auc_tr:.4f} | test AUC: {results['DKT']['auc']:.4f} | "
          f"gap: {auc_tr - results['DKT']['auc']:.4f}")

    # ============================================================
    # 5. GKT
    # ============================================================
    print("\n=== 4. GKT ===")
    gkt = GKT(K, hidden=H, gnn_layers=2, dropout=0.4)
    train_model(gkt, train_t, A_t, U_t, epochs=EPOCHS,
                batch_size=args.batch_size, weight_decay=5e-3,
                is_graph_model=True)
    p_tr, y_tr, m_tr = predict(gkt, train_t, A_t, U_t, is_graph_model=True)
    auc_tr = binary_metrics(y_tr, p_tr, m_tr)["auc"]
    p, y, m = predict(gkt, test_t, A_t, U_t, is_graph_model=True)
    results["GKT"] = binary_metrics(y, p, m)
    results["GKT"]["ece"] = expected_calibration_error(
        y[m.astype(bool)], p[m.astype(bool)]
    )
    print(f"GKT train AUC: {auc_tr:.4f} | test AUC: {results['GKT']['auc']:.4f} | "
          f"gap: {auc_tr - results['GKT']['auc']:.4f}")

    # ============================================================
    # 6. UAG-KT
    # ============================================================
    print("\n=== 5. UAG-KT ===")
    uag = UAGKT(K, hidden=H, gnn_layers=2, dropout=0.4)
    train_model(uag, train_t, A_t, U_t, epochs=EPOCHS,
                batch_size=args.batch_size, weight_decay=5e-3,
                is_graph_model=True)
    p_tr, y_tr, m_tr = predict(uag, train_t, A_t, U_t, is_graph_model=True)
    auc_tr = binary_metrics(y_tr, p_tr, m_tr)["auc"]
    p, y, m = predict(uag, test_t, A_t, U_t, is_graph_model=True)
    results["UAG-KT"] = binary_metrics(y, p, m)
    results["UAG-KT"]["ece"] = expected_calibration_error(
        y[m.astype(bool)], p[m.astype(bool)]
    )
    mc = uag.predict_with_uncertainty(
        test_t["concepts"], test_t["responses"], A_t, U_t, n_samples=10
    )
    mask_np = test_t["mask"].numpy()
    prob_var = mc["prob_var"].numpy()
    results["UAG-KT"]["mean_uncertainty"] = float(
        (prob_var * mask_np).sum() / mask_np.sum()
    )
    print(f"UAG-KT train AUC: {auc_tr:.4f} | test AUC: {results['UAG-KT']['auc']:.4f} | "
          f"gap: {auc_tr - results['UAG-KT']['auc']:.4f}")

    # ============================================================
    # 7. Summary
    # ============================================================
    print("\n=== SUMMARY ===")
    keys = ["auc", "brier", "accuracy", "ece"]
    print(f"{'model':10s} " + " ".join(f"{k:>10s}" for k in keys))
    for mname, r in results.items():
        print(f"{mname:10s} " + " ".join(
            f"{r.get(k, float('nan')):10.4f}" for k in keys
        ))

    print("\n=== SANITY CHECK ===")
    dkt_auc = results["DKT"]["auc"]
    gkt_auc = results["GKT"]["auc"]
    uag_auc = results["UAG-KT"]["auc"]
    dkt_ece = results["DKT"]["ece"]
    uag_ece = results["UAG-KT"]["ece"]
    bkt_auc = results["BKT"]["auc"]

    checks = [
        (f"DKT AUC > BKT AUC ({dkt_auc:.3f} > {bkt_auc:.3f})",
         dkt_auc > bkt_auc),
        (f"GKT >= DKT ({gkt_auc:.3f} >= {dkt_auc:.3f})",
         gkt_auc >= dkt_auc - 0.01),
        (f"UAG-KT >= DKT ({uag_auc:.3f} >= {dkt_auc:.3f})",
         uag_auc >= dkt_auc - 0.01),
        (f"UAG-KT ECE <= DKT ECE ({uag_ece:.4f} <= {dkt_ece:.4f})",
         uag_ece <= dkt_ece + 0.02),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    # ============================================================
    # 8. Save JSON 
    # ============================================================
    if args.output_json:
        import json
        from pathlib import Path
        def _json_safe(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, (np.floating, np.integer)):
                    out[k] = float(v)
                elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    out[k] = None
                else:
                    out[k] = v
            return out
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({k: _json_safe(v) for k, v in results.items()},
                      f, indent=2)
        print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()