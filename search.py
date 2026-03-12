"""
search.py — two-stage retrieval pipeline:
  1. Dense ANN search via FAISS HNSW
  2. (Optional) Cross-encoder reranking
  3. Group chunks by paper and return the top-N papers
"""

import time
import random
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pandas as pd

from config import Config


# ---------------------------------------------------------------------------
# Reranker loader
# ---------------------------------------------------------------------------

def load_reranker(cfg: Config) -> Optional[CrossEncoder]:
    if not cfg.use_reranker:
        return None
    print(f"Loading reranker: {cfg.rerank_model_name}")
    reranker = CrossEncoder(cfg.rerank_model_name)
    print("Reranker loaded.")
    return reranker


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search_papers(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.IndexHNSWFlat,
    chunks_df: pd.DataFrame,
    cfg: Config,
    reranker: Optional[CrossEncoder] = None,
    retrieve_k: Optional[int] = None,
    final_papers: Optional[int] = None,
    chunks_per_paper: int = 3,
) -> List[Dict]:
    """
    Search for papers relevant to `query`.

    Steps
    -----
    1. Encode query with the bi-encoder.
    2. Retrieve `retrieve_k` candidate chunks via FAISS ANN.
    3. Optionally rerank with a cross-encoder.
    4. Group chunks by paper; aggregate by best chunk score.
    5. Return the top `final_papers` papers.
    """
    if retrieve_k is None:
        retrieve_k = cfg.retrieve_k
    if final_papers is None:
        final_papers = cfg.final_papers

    # 1. Encode query
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

    # 2. ANN retrieval
    scores, idxs = index.search(q_emb, retrieve_k)
    scores, idxs = scores[0], idxs[0]

    # 3. Optional cross-encoder reranking
    if reranker is not None:
        pairs    = [(query, chunks_df.loc[int(i), "chunk_text"]) for i in idxs]
        r_scores = reranker.predict(pairs)
        order    = np.argsort(-np.array(r_scores))
        idxs     = idxs[order]
        scores   = np.array(r_scores)[order].astype("float32")

    # 4. Group by paper
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for s, i in zip(scores, idxs):
        row = chunks_df.loc[int(i)]
        grouped[row["paper_id"]].append({
            "score":    float(s),
            "chunk_id": int(row["chunk_id"]),
            "snippet":  row["chunk_text"][:500] + (
                "..." if len(row["chunk_text"]) > 500 else ""
            ),
        })

    # 5. Build paper-level results
    results = []
    for pid, chunks in grouped.items():
        chunks_sorted = sorted(chunks, key=lambda x: -x["score"])
        top_chunks    = chunks_sorted[:chunks_per_paper]
        row0          = chunks_df[chunks_df["paper_id"] == pid].iloc[0]
        abstract      = str(row0["abstract"])
        results.append({
            "paper_id":   pid,
            "title":      row0["title"],
            "url":        row0["url"],
            "best_score": top_chunks[0]["score"],
            "abstract":   abstract[:400] + ("..." if len(abstract) > 400 else ""),
            "chunks":     top_chunks,
        })

    results = sorted(results, key=lambda x: -x["best_score"])[:final_papers]
    return results


# ---------------------------------------------------------------------------
# Similar-paper lookup (vector-space neighbour)
# ---------------------------------------------------------------------------

def similar_papers(
    result_item: Dict,
    index: faiss.IndexHNSWFlat,
    chunks_df: pd.DataFrame,
    k: int = 10,
) -> List[Dict]:
    """Return papers whose best chunk is nearest to the top chunk of result_item."""
    pid = result_item["paper_id"]
    cid = result_item["chunks"][0]["chunk_id"]

    matches = chunks_df.index[
        (chunks_df["paper_id"] == pid) & (chunks_df["chunk_id"] == cid)
    ]
    if len(matches) == 0:
        return []

    row_idx = int(matches[0])
    v       = np.array(index.reconstruct(row_idx), dtype="float32").reshape(1, -1)
    scores, idxs = index.search(v, k * 8)
    scores, idxs = scores[0], idxs[0]

    grouped = {}
    for s, i in zip(scores, idxs):
        row  = chunks_df.loc[int(i)]
        pid2 = row["paper_id"]
        if pid2 == pid:
            continue
        if pid2 not in grouped:
            grouped[pid2] = {
                "paper_id": pid2,
                "title":    row["title"],
                "score":    float(s),
                "url":      row["url"],
            }
        if len(grouped) >= k:
            break

    return list(grouped.values())


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark(
    queries: List[str],
    embedder: SentenceTransformer,
    index: faiss.IndexHNSWFlat,
    chunks_df: pd.DataFrame,
    cfg: Config,
    reranker: Optional[CrossEncoder] = None,
    runs: int = 20,
) -> Dict:
    times = []
    for _ in range(runs):
        query = random.choice(queries)
        t0    = time.time()
        search_papers(query, embedder, index, chunks_df, cfg, reranker=reranker)
        times.append((time.time() - t0) * 1000.0)

    times = np.array(times)
    return {
        "p50_ms":  float(np.percentile(times, 50)),
        "p95_ms":  float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "runs":    runs,
        "rerank":  reranker is not None,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def pretty_print(results: List[Dict]) -> None:
    sep = "=" * 90
    for r in results:
        print(sep)
        print(f"Score : {r['best_score']:.4f}")
        print(f"Title : {r['title']}")
        print(f"ID    : {r['paper_id']}")
        if r["url"]:
            print(f"URL   : {r['url']}")
        print(f"Abstract: {r['abstract']}")
        print("\nTop Supporting Chunks:")
        for c in r["chunks"]:
            print(f"  • [Chunk {c['chunk_id']} | Score {c['score']:.4f}]")
            print(f"    {c['snippet']}")
        print()
