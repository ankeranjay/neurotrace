"""Pipeline end-to-end dengan fix analisis:
  - U = zeros (graph confidence penuh, no noise)
  - H = 48, dropout = 0.4
  - BCE only (no gaussian_nll)
  - train AUC untuk diagnosis overfit
"""
from __future__ import annotations
import sys
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


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    K, H, GNN_L = 8, 48, 2      # H kecil untuk dataset kecil
    N_STEPS = 40
    EPOCHS = 80

    print("=== 1. Generate synthetic data ===")
    mastery, sequences, A_np = generate(
        n_students=500, n_concepts=K, n_steps=N_STEPS
    )
    tensors = to_tensors(sequences, max_concepts=K, max_len=N_STEPS)
    print(f"Sequences: {len(sequences)} | max_len: {N_STEPS}")

    # Student-level split
    n = len(sequences)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    def slice_t(d, ids):
        return {k: torch.from_numpy(v[ids]) for k, v in d.items()}

    train_t = slice_t(tensors, tr_idx)
    test_t = slice_t(tensors, te_idx)

    # ============================================================
    # Graph: pakai A_np (real chain), U = 0 (no random noise)
    # ============================================================
    kg = KnowledgeGraph([str(i) for i in range(K)])
    for i in range(K):
        for j in range(K):
            if A_np[i, j] > 0:
                kg.add_edge(str(i), str(j), weight=float(A_np[i, j]), confidence=1.0)
    A_t, U_t = kg.to_tensors()
    U_t = torch.zeros_like(U_t)     # confidence penuh, no noise
    print(f"Graph edges: {int((A_np > 0).sum())} | U mean: {float(U_t.mean()):.3f}")

    results = {}

    # ============================================================
    # BKT baseline
    # ============================================================
    print("\n=== 2. BKT baseline ===")
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
    results["BKT"] = binary_metrics(y_true, y_prob)
    results["BKT"]["ece"] = expected_calibration_error(
        np.array(y_true, dtype=bool), np.array(y_prob)
    )
    print(results["BKT"])

    # ============================================================
    # DKT — dengan train/test AUC diagnosis
    # ============================================================
    print("\n=== 3. DKT ===")
    dkt = DKT(K, hidden=H, dropout=0.4)
    train_model(dkt, train_t, A_t, U_t, epochs=EPOCHS,
                weight_decay=5e-3, is_graph_model=False)

    p_tr, y_tr, m_tr = predict(dkt, train_t, A_t, U_t, is_graph_model=False)
    auc_tr = binary_metrics(y_tr, p_tr, m_tr)["auc"]

    p, y, m = predict(dkt, test_t, A_t, U_t, is_graph_model=False)
    results["DKT"] = binary_metrics(y, p, m)
    results["DKT"]["ece"] = expected_calibration_error(
        y[m.astype(bool)], p[m.astype(bool)]
    )
    print(f"DKT train AUC: {auc_tr:.4f} | test AUC: {results['DKT']['auc']:.4f}")
    print(f"DKT gap (overfit check): {auc_tr - results['DKT']['auc']:.4f}")

    # ============================================================
    # GKT
    # ============================================================
    print("\n=== 4. GKT ===")
    gkt = GKT(K, hidden=H, gnn_layers=GNN_L, dropout=0.4)
    train_model(gkt, train_t, A_t, U_t, epochs=EPOCHS,
                weight_decay=5e-3, is_graph_model=True)

    p_tr, y_tr, m_tr = predict(gkt, train_t, A_t, U_t, is_graph_model=True)
    auc_tr = binary_metrics(y_tr, p_tr, m_tr)["auc"]

    p, y, m = predict(gkt, test_t, A_t, U_t, is_graph_model=True)
    results["GKT"] = binary_metrics(y, p, m)
    results["GKT"]["ece"] = expected_calibration_error(
        y[m.astype(bool)], p[m.astype(bool)]
    )
    print(f"GKT train AUC: {auc_tr:.4f} | test AUC: {results['GKT']['auc']:.4f}")
    print(f"GKT gap: {auc_tr - results['GKT']['auc']:.4f}")

    # ============================================================
    # UAG-KT
    # ============================================================
    print("\n=== 5. UAG-KT ===")
    uag = UAGKT(K, hidden=H, gnn_layers=GNN_L, dropout=0.4)
    train_model(uag, train_t, A_t, U_t, epochs=EPOCHS,
                weight_decay=5e-3, is_graph_model=True)

    p_tr, y_tr, m_tr = predict(uag, train_t, A_t, U_t, is_graph_model=True)
    auc_tr = binary_metrics(y_tr, p_tr, m_tr)["auc"]

    p, y, m = predict(uag, test_t, A_t, U_t, is_graph_model=True)
    results["UAG-KT"] = binary_metrics(y, p, m)
    results["UAG-KT"]["ece"] = expected_calibration_error(
        y[m.astype(bool)], p[m.astype(bool)]
    )

    # MC Dropout (masked)
    mc = uag.predict_with_uncertainty(
        test_t["concepts"], test_t["responses"], A_t, U_t, n_samples=15
    )
    mask_np = test_t["mask"].numpy()
    prob_var = mc["prob_var"].numpy()
    results["UAG-KT"]["mean_uncertainty"] = float(
        (prob_var * mask_np).sum() / mask_np.sum()
    )
    print(f"UAG-KT train AUC: {auc_tr:.4f} | test AUC: {results['UAG-KT']['auc']:.4f}")
    print(f"UAG-KT gap: {auc_tr - results['UAG-KT']['auc']:.4f}")

    # ============================================================
    # Summary
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
        (f"DKT AUC > BKT AUC ({dkt_auc:.3f} > {bkt_auc:.3f})", dkt_auc > bkt_auc),
        ("GKT >= DKT (graph membantu)", gkt_auc >= dkt_auc - 0.01),
        ("UAG-KT >= DKT (uncertainty membantu)", uag_auc >= dkt_auc - 0.01),
        ("UAG-KT ECE <= DKT ECE (kalibrasi)", uag_ece <= dkt_ece + 0.02),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()