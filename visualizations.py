# ===========================================================================
# VISUALIZATION CELLS — paste these at the end of your Colab notebook
# Requires: cfg, chunks_df, ds, ds_f, index, embedder, reranker
# All plots are saved to Google Drive under cfg.out_dir/plots/
# ===========================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import time, random

PLOT_DIR = os.path.join(cfg.out_dir, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Shared style
COLORS = {
    "dense":   "#4F8EF7",
    "rerank":  "#F76B4F",
    "before":  "#A8D5A2",
    "after":   "#3A7D44",
    "chunks":  "#7B68EE",
    "scores":  "#F4A261",
}
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. DATASET SIZE — before vs after ML filter
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

labels = ["Raw dataset", "After ML filter"]
values = [len(ds), len(ds_f)]
bar_colors = [COLORS["before"], COLORS["after"]]

bars = ax.bar(labels, values, color=bar_colors, width=0.45, zorder=3)

for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(values) * 0.01,
        f"{v:,}",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

reduction = (1 - values[1] / values[0]) * 100
ax.set_title(
    f"Dataset size before vs after ML keyword filter\n"
    f"({reduction:.1f}% of non-ML papers removed)",
    fontsize=12, pad=12
)
ax.set_ylabel("Number of papers")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, max(values) * 1.15)

save(fig, "01_dataset_filter.png")


# ---------------------------------------------------------------------------
# 2. CHUNKS-PER-PAPER DISTRIBUTION
# ---------------------------------------------------------------------------

chunk_counts = chunks_df.groupby("paper_id")["chunk_id"].count()

fig, ax = plt.subplots(figsize=(8, 4))

bins = range(1, chunk_counts.max() + 2)
ax.hist(chunk_counts, bins=bins, color=COLORS["chunks"], edgecolor="white",
        linewidth=0.6, zorder=3, align="left")

ax.axvline(chunk_counts.mean(), color="#333", linestyle="--", linewidth=1.4,
           label=f"Mean: {chunk_counts.mean():.1f}")
ax.axvline(chunk_counts.median(), color="#888", linestyle=":", linewidth=1.4,
           label=f"Median: {chunk_counts.median():.1f}")

ax.set_title("Chunks per paper distribution", fontsize=12, pad=12)
ax.set_xlabel("Number of chunks")
ax.set_ylabel("Number of papers")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(frameon=False)

save(fig, "02_chunks_per_paper.png")


# ---------------------------------------------------------------------------
# 3. SCORE DISTRIBUTION — dense vs reranked, across multiple queries
# ---------------------------------------------------------------------------

bench_queries = [
    "diffusion transformer DiT text-to-image",
    "vision transformer patch size ablation",
    "contrastive learning sentence embeddings",
    "reinforcement learning reward shaping",
    "large language model fine-tuning RLHF",
]

dense_scores_all   = []
rerank_scores_all  = []

for q in bench_queries:
    q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, cfg.retrieve_k)
    dense_scores_all.extend(scores[0].tolist())

    if reranker is not None:
        pairs    = [(q, chunks_df.loc[int(i), "chunk_text"]) for i in idxs[0]]
        r_scores = reranker.predict(pairs)
        rerank_scores_all.extend(r_scores.tolist())

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

axes[0].hist(dense_scores_all, bins=50, color=COLORS["dense"],
             edgecolor="white", linewidth=0.5, zorder=3)
axes[0].set_title("Dense retrieval score distribution\n(FAISS inner product)", fontsize=11)
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Count")

if rerank_scores_all:
    axes[1].hist(rerank_scores_all, bins=50, color=COLORS["rerank"],
                 edgecolor="white", linewidth=0.5, zorder=3)
    axes[1].set_title("Cross-encoder reranker score distribution", fontsize=11)
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Count")
else:
    axes[1].text(0.5, 0.5, "Reranker disabled", ha="center", va="center",
                 transform=axes[1].transAxes, fontsize=12)
    axes[1].set_title("Reranker score distribution", fontsize=11)

fig.suptitle(
    f"Score distributions across {len(bench_queries)} queries · {cfg.retrieve_k} candidates each",
    fontsize=12, y=1.02
)
plt.tight_layout()
save(fig, "03_score_distributions.png")


# ---------------------------------------------------------------------------
# 4. LATENCY — dense vs reranked (violin + strip)
# ---------------------------------------------------------------------------

def collect_latencies(queries, runs, use_rerank):
    times = []
    for _ in range(runs):
        q  = random.choice(queries)
        t0 = time.time()
        q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(q_emb, cfg.retrieve_k)
        if use_rerank and reranker is not None:
            pairs = [(q, chunks_df.loc[int(i), "chunk_text"]) for i in idxs[0]]
            reranker.predict(pairs)
        times.append((time.time() - t0) * 1000)
    return times

print("Collecting latency samples (dense)...")
dense_times  = collect_latencies(bench_queries, runs=30, use_rerank=False)
print("Collecting latency samples (reranked)...")
rerank_times = collect_latencies(bench_queries, runs=15, use_rerank=True)

fig, ax = plt.subplots(figsize=(8, 5))

data   = [dense_times, rerank_times]
labels = ["Dense only", "Dense + Rerank"]
colors = [COLORS["dense"], COLORS["rerank"]]

parts = ax.violinplot(data, positions=[1, 2], widths=0.5,
                      showmedians=True, showextrema=False)

for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
parts["cmedians"].set_color("#222")
parts["cmedians"].set_linewidth(2)

# Overlay individual points
for i, (d, color) in enumerate(zip(data, colors), start=1):
    jitter = np.random.uniform(-0.08, 0.08, size=len(d))
    ax.scatter(np.full(len(d), i) + jitter, d,
               color=color, alpha=0.5, s=18, zorder=3)

# Annotate p50 / p95
for i, d in enumerate(data, start=1):
    p50 = np.percentile(d, 50)
    p95 = np.percentile(d, 95)
    ax.text(i + 0.28, p50,  f"p50: {p50:.0f}ms",  va="center", fontsize=9)
    ax.text(i + 0.28, p95,  f"p95: {p95:.0f}ms",  va="center", fontsize=9, color="#888")

ax.set_xticks([1, 2])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Latency (ms)")
ax.set_title("Query latency: dense retrieval vs dense + reranking", fontsize=12, pad=12)
ax.set_xlim(0.5, 2.8)

save(fig, "04_latency_violin.png")

print(f"\n✅ All plots saved to: {PLOT_DIR}")
