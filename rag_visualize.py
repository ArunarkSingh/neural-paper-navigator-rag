"""
rag_visualize.py — Visualisations for RAG evaluation results.

Generates:
  - Bar chart comparing dense vs hybrid on all three RAGAS metrics
  - Per-query answer comparison table printed to stdout
  - Results saved as CSV + JSON to the output directory
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


COLORS = {
    "dense":  "#4F8EF7",
    "hybrid": "#F76B4F",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def plot_ragas_comparison(
    dense_result: Dict,
    hybrid_result: Dict,
    eval_queries: List[str],
    out_dir: str,
    groq_model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Side-by-side bar chart: dense vs hybrid for faithfulness,
    answer relevancy, and context precision.
    Returns the path to the saved figure.
    """
    metrics_names  = ["faithfulness", "answer_relevancy", "context_precision"]
    metrics_labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision"]

    dense_scores  = [dense_result[m]  for m in metrics_names]
    hybrid_scores = [hybrid_result[m] for m in metrics_names]

    x     = np.arange(len(metrics_labels))
    width = 0.33

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width / 2, dense_scores,  width,
                   label="Dense-only",          color=COLORS["dense"],  zorder=3)
    bars2 = ax.bar(x + width / 2, hybrid_scores, width,
                   label="Hybrid (BM25 + Dense)", color=COLORS["hybrid"], zorder=3)

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylabel("Score (0–1)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"RAGAS Evaluation: Dense-only vs Hybrid Retrieval\n"
        f"({len(eval_queries)} queries · Groq {groq_model} judge)",
        fontsize=12, pad=14,
    )
    ax.legend(frameon=False, fontsize=10)

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "ragas_comparison.png")
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Saved: {plot_path}")
    plt.show()
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# Comparison table (printed + saved to CSV)
# ---------------------------------------------------------------------------

def print_comparison_table(
    dense_result: Dict,
    hybrid_result: Dict,
    out_dir: str,
) -> pd.DataFrame:
    metrics_names = ["faithfulness", "answer_relevancy", "context_precision"]

    comparison = pd.DataFrame({
        "Metric":     metrics_names,
        "Dense-only": [round(dense_result[m],  4) for m in metrics_names],
        "Hybrid":     [round(hybrid_result[m], 4) for m in metrics_names],
    })
    comparison["Delta (Hybrid − Dense)"] = (
        comparison["Hybrid"] - comparison["Dense-only"]
    ).round(4)
    comparison["Winner"] = comparison["Delta (Hybrid − Dense)"].apply(
        lambda d: "Hybrid ✓" if d > 0.01 else ("Dense ✓" if d < -0.01 else "Tie")
    )

    print("\n" + "=" * 68)
    print("RAGAS EVALUATION: Dense-only vs Hybrid Retrieval")
    print("=" * 68)
    print(comparison.to_string(index=False))
    print("=" * 68)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ragas_comparison.csv")
    comparison.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table: {csv_path}")

    return comparison


# ---------------------------------------------------------------------------
# Per-query answer comparison
# ---------------------------------------------------------------------------

def print_per_query_comparison(
    dense_outputs: List[Dict],
    hybrid_outputs: List[Dict],
) -> None:
    print("\nPER-QUERY ANSWER COMPARISON\n" + "=" * 80)
    for i, (d, h) in enumerate(zip(dense_outputs, hybrid_outputs)):
        print(f"\nQ{i+1}: {d['question']}")
        print(f"  Retrieval → Dense: {d['retrieval_ms']}ms | Hybrid: {h['retrieval_ms']}ms")
        print(f"\n  [DENSE]  {d['answer'][:300]}...")
        print(f"\n  [HYBRID] {h['answer'][:300]}...")
        print("-" * 80)


# ---------------------------------------------------------------------------
# Save full results JSON
# ---------------------------------------------------------------------------

def save_results_json(
    eval_queries: List[str],
    dense_result: Dict,
    hybrid_result: Dict,
    dense_outputs: List[Dict],
    hybrid_outputs: List[Dict],
    out_dir: str,
    groq_model: str,
    embed_model: str,
    rerank_model: str,
) -> str:
    metrics_names = ["faithfulness", "answer_relevancy", "context_precision"]

    full_results = {
        "eval_queries":  eval_queries,
        "dense_ragas":   {m: float(dense_result[m])  for m in metrics_names},
        "hybrid_ragas":  {m: float(hybrid_result[m]) for m in metrics_names},
        "dense_outputs":  dense_outputs,
        "hybrid_outputs": hybrid_outputs,
        "run_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "groq_model":     groq_model,
        "judge_model":    groq_model,
        "embed_model":    embed_model,
        "rerank_model":   rerank_model,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "rag_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Full results saved: {json_path}")
    return json_path
