"""Evaluasi performa model pada 5 langkah pertama siswa (Cold-Start)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.data.assistments_hf import load_assistments_hf, build_cooccurrence_graph
from src.models.bkt_per_skill import BKTPerSkill
from src.models.dkt import DKT
from src.models.gkt import GKT
from src.models.neurotrace import UAGKT
from src.training.train import train_model, predict
from src.evaluation.metrics import binary_metrics
from src.knowledge_graph.graph import KnowledgeGraph

def main():
    torch.manual_seed(42); np.random.seed(42)
    print("Loading full dataset...")
    sequences, K, _ = load_assistments_hf(n_students=9999, max_len=100, verbose=True)
    
    from src.data.synthetic import to_tensors
    tensors = to_tensors(sequences, max_concepts=K, max_len=100)
    n = len(sequences); idx = np.random.permutation(n); split = int(0.8 * n)
    
    def slice_t(d, ids): return {k: torch.from_numpy(v[ids]) for k, v in d.items()}
    train_t = slice_t(tensors, idx[:split]); test_t = slice_t(tensors, idx[split:])
    
    A_np = build_cooccurrence_graph(sequences, K)
    kg = KnowledgeGraph([str(i) for i in range(K)])
    for i in range(K):
        for j in range(K):
            if A_np[i, j] > 0: kg.add_edge(str(i), str(j), weight=float(A_np[i, j]))
    A_t, U_t = kg.to_tensors(); U_t = torch.zeros_like(U_t)
    
    print("Training models (50 epochs)...")
    dkt = DKT(K, hidden=48, dropout=0.4); train_model(dkt, train_t, A_t, U_t, epochs=50, is_graph_model=False)
    gkt = GKT(K, hidden=48, dropout=0.4); train_model(gkt, train_t, A_t, U_t, epochs=50, is_graph_model=True)
    uag = UAGKT(K, hidden=48, dropout=0.4); train_model(uag, train_t, A_t, U_t, epochs=50, is_graph_model=True)
    
    # Prediksi di test set
    p_dkt, y, m = predict(dkt, test_t, A_t, U_t, is_graph_model=False)
    p_gkt, _, _ = predict(gkt, test_t, A_t, U_t, is_graph_model=True)
    p_uag, _, _ = predict(uag, test_t, A_t, U_t, is_graph_model=True)
    
    # Hitung AUC untuk 1..5 langkah pertama
    print("\n=== COLD-START AUC (Step 1-5) ===")
    print(f"{'Step':<5} | {'DKT':<8} | {'GKT':<8} | {'UAG-KT':<8}")
    print("-" * 35)
    for step in range(5):
        mask = (m[:, step] > 0)
        if mask.sum() == 0: continue
        auc_dkt = binary_metrics(y[mask, step], p_dkt[mask, step])["auc"]
        auc_gkt = binary_metrics(y[mask, step], p_gkt[mask, step])["auc"]
        auc_uag = binary_metrics(y[mask, step], p_uag[mask, step])["auc"]
        print(f"{step+1:<5} | {auc_dkt:.4f}   | {auc_gkt:.4f}   | {auc_uag:.4f}")

if __name__ == "__main__":
    main()