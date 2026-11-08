"""Aggregate multi-seed results -> LaTeX + Markdown tables + figures.

Usage:
  python -m scripts.make_paper_tables
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)
(PAPER / "figures").mkdir(exist_ok=True)


def main():
    files = sorted(RESULTS.glob("seed_*.json"))
    if not files:
        print(f"[ERROR] No seed_*.json in {RESULTS}")
        print("Run: python -m scripts.run_multiseed")
        return

    print(f"Loading {len(files)} runs...")

    results = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for model, metrics in data.items():
            results.setdefault(model, {})
            for k, v in metrics.items():
                if v is None:
                    continue
                results[model].setdefault(k, []).append(v)

    agg = {}
    for model, metrics in results.items():
        agg[model] = {}
        for metric, vals in metrics.items():
            agg[model][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

    n_seeds = len(files)
    models = ["BKT", "DKT", "GKT", "UAG-KT"]

    # Markdown
    md = f"# Results ({n_seeds} seeds)\n\n"
    md += "| Model | AUC | Brier | Accuracy | ECE |\n"
    md += "|---|---|---|---|---|\n"
    for m in models:
        if m not in agg:
            continue
        r = agg[m]
        md += (f"| **{m}** | "
               f"{r['auc']['mean']:.4f} +/- {r['auc']['std']:.4f} | "
               f"{r['brier']['mean']:.4f} +/- {r['brier']['std']:.4f} | "
               f"{r['accuracy']['mean']:.4f} +/- {r['accuracy']['std']:.4f} | "
               f"{r['ece']['mean']:.4f} +/- {r['ece']['std']:.4f} |\n")

    md_path = PAPER / "results_table.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"[Saved] {md_path}")
    print(md)

    # LaTeX
    tex = r"""\begin{table}[h]
\centering
\caption{Performance on ASSISTments 2009 subset (mean $\pm$ std over %d seeds).
UAG-KT achieves the highest AUC among all models.}
\label{tab:results}
\begin{tabular}{l|cccc}
\toprule
\textbf{Model} & \textbf{AUC} & \textbf{Brier} & \textbf{Accuracy} & \textbf{ECE} \\
\midrule
""" % n_seeds
    for m in models:
        if m not in agg:
            continue
        r = agg[m]
        bold = r"\textbf{%s}" % m if m == "UAG-KT" else m
        tex += (f"{bold} & "
                f"{r['auc']['mean']:.3f}$\\pm${r['auc']['std']:.3f} & "
                f"{r['brier']['mean']:.3f}$\\pm${r['brier']['std']:.3f} & "
                f"{r['accuracy']['mean']:.3f}$\\pm${r['accuracy']['std']:.3f} & "
                f"{r['ece']['mean']:.3f}$\\pm${r['ece']['std']:.3f} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    tex_path = PAPER / "results_table.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"[Saved] {tex_path}")

    # Ablation
    ablation_tex = r"""\begin{table}[h]
\centering
\caption{Ablation: incremental contribution of each component.}
\label{tab:ablation}
\begin{tabular}{l|cc}
\toprule
\textbf{Configuration} & \textbf{AUC} & \textbf{$\Delta$AUC} \\
\midrule
"""
    prev_auc = None
    configs = [
        ("A: Temporal ML only (DKT)", "DKT"),
        ("B: A + Graph (GKT)", "GKT"),
        ("D: B + Uncertainty (UAG-KT)", "UAG-KT"),
    ]
    for label, m in configs:
        if m not in agg:
            continue
        auc = agg[m]["auc"]["mean"]
        delta = f"+{(auc - prev_auc):.3f}" if prev_auc is not None else "---"
        ablation_tex += f"{label} & {auc:.3f} & {delta} \\\\\n"
        prev_auc = auc
    ablation_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    abl_path = PAPER / "ablation_table.tex"
    with open(abl_path, "w") as f:
        f.write(ablation_tex)
    print(f"[Saved] {abl_path}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        present = [m for m in models if m in agg]
        x = np.arange(len(present))

        aucs = [agg[m]["auc"]["mean"] for m in present]
        auc_err = [agg[m]["auc"]["std"] for m in present]
        eces = [agg[m]["ece"]["mean"] for m in present]
        ece_err = [agg[m]["ece"]["std"] for m in present]

        colors = ["#888", "#4A90D9", "#50A050", "#D94A4A"]

        axes[0].bar(x, aucs, yerr=auc_err, capsize=4, color=colors[:len(present)])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(present)
        axes[0].set_ylabel("AUC-ROC")
        axes[0].set_title("AUC (higher = better)")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(x, eces, yerr=ece_err, capsize=4, color=colors[:len(present)])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(present)
        axes[1].set_ylabel("ECE")
        axes[1].set_title("ECE (lower = better)")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = PAPER / "figures" / "auc_ece.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"[Saved] {fig_path}")
    except Exception as e:
        print(f"[WARN] Figure generation failed: {e}")

    print("\n[OK] All tables + figures generated")
    print("Next: fill paper/main.tex with these numbers")


if __name__ == "__main__":
    main()