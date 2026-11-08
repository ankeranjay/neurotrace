"""Plot reliability diagram (calibration curve) for DKT vs UAG-KT."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.data.assistments_hf import load_assistments_hf, build_cooccurrence_graph
from src.models.dkt import DKT
from src.models.neurotrace import UAGKT
from src.training.train import train_model, predict
from src.knowledge_graph.graph import KnowledgeGraph

def plot_reliability(y_true, y_prob, model_name, ax):
    bins = np.linspace(0, 1, 11)
    binids = np.digitize(y_prob, bins) - 1
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
    
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", label=model_name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title(model_name)
    ax.legend()

def main():
    torch.manual_seed(42); np.random.seed(42)
    sequences, K, _ = load_assistments_hf(n_students=9999, max_len=100, verbose=False)
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

    print("Training models...")
    dkt = DKT(K, hidden=48, dropout=0.4); train_model(dkt, train_t, A_t, U_t, epochs=50, is_graph_model=False)
    uag = UAGKT(K, hidden=48, dropout=0.4); train_model(uag, train_t, A_t, U_t, epochs=50, is_graph_model=True)
    
    p_dkt, y, m = predict(dkt, test_t, A_t, U_t, is_graph_model=False)
    p_uag, _, _ = predict(uag, test_t, A_t, U_t, is_graph_model=True)

    mask = m.astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_reliability(y[mask], p_dkt[mask], "DKT", axes[0])
    plot_reliability(y[mask], p_uag[mask], "UAG-KT (Ours)", axes[1])
    plt.tight_layout()
    plt.savefig("paper/figures/reliability_diagram.png", dpi=150)
    print("[Saved] paper/figures/reliability_diagram.png")

if __name__ == "__main__":
    main()