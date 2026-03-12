"""
rag_retrieval.py — Hybrid retrieval for the RAG pipeline.

Adds BM25 sparse retrieval on top of the existing FAISS dense index,
fused via Reciprocal Rank Fusion (RRF).
"""

import re
import time
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

from rank_bm25 import BM25Okapi

from config import Config


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_bm25_index(chunks_df) -> BM25Okapi:
    """
    Build a BM25 index over chunk_text (not the enriched 'text' field,
    so scores reflect actual content rather than the title prefix added
    for dense embedding).
    """
    print("Building BM25 index...")
    t0 = time.time()
    corpus_tokens = [simple_tokenize(t) for t in chunks_df["chunk_text"].tolist()]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"BM25 index built over {len(corpus_tokens):,} chunks in {time.time()-t0:.1f}s")
    return bm25


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def bm25_retrieve(query: str, bm25: BM25Okapi, top_k: int) -> List[Tuple[int, float]]:
    """Returns [(chunk_index, bm25_score), ...] sorted descending."""
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idxs = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idxs]


def dense_retrieve(query: str, embedder, index, top_k: int) -> List[Tuple[int, float]]:
    """Returns [(chunk_index, cosine_score), ...] sorted descending."""
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0])]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists using RRF.
    Each list is [(chunk_idx, score), ...] sorted by descending score.
    Returns fused list sorted by descending RRF score.

    RRF formula: score(d) = Σ 1 / (k + rank(d))
    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    """
    rrf_scores: Dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_idx, _) in enumerate(ranked):
            rrf_scores[chunk_idx] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: -x[1])


def hybrid_retrieve(
    query: str,
    embedder,
    index,
    bm25: BM25Okapi,
    retrieve_k: int = 100,
    rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    """BM25 + dense retrieval fused with RRF. Returns top retrieve_k chunks."""
    dense_results = dense_retrieve(query, embedder, index, retrieve_k)
    bm25_results  = bm25_retrieve(query, bm25, retrieve_k)
    fused         = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k)
    return fused[:retrieve_k]


# ---------------------------------------------------------------------------
# search_papers_rag — full retrieval + rerank + group by paper
# ---------------------------------------------------------------------------

def search_papers_rag(
    query: str,
    embedder,
    index,
    bm25: BM25Okapi,
    chunks_df,
    cfg: Config,
    reranker=None,
    retrieval_mode: str = "hybrid",
    retrieve_k: int = 100,
    final_papers: int = 5,
    chunks_per_paper: int = 2,
) -> Tuple[List[Dict], List[str]]:
    """
    Full retrieval pipeline for RAG.

    Args:
        retrieval_mode: "hybrid" (BM25 + dense RRF) or "dense" (FAISS only)

    Returns:
        results  — list of paper dicts with title, url, abstract, chunks
        contexts — flat list of top chunk texts (fed directly to the LLM)
    """
    # 1. Retrieve
    if retrieval_mode == "hybrid":
        ranked = hybrid_retrieve(query, embedder, index, bm25, retrieve_k=retrieve_k)
        idxs   = np.array([r[0] for r in ranked])
        scores = np.array([r[1] for r in ranked], dtype="float32")
    else:
        q_emb          = embedder.encode([query], normalize_embeddings=True).astype("float32")
        sc, ix         = index.search(q_emb, retrieve_k)
        idxs, scores   = ix[0], sc[0]

    # 2. Optional cross-encoder reranking
    if reranker is not None:
        pairs    = [(query, chunks_df.loc[int(i), "chunk_text"]) for i in idxs]
        r_scores = reranker.predict(pairs)
        order    = np.argsort(-np.array(r_scores))
        idxs     = idxs[order]
        scores   = np.array(r_scores)[order].astype("float32")

    # 3. Group chunks by paper, keep top-scoring chunks per paper
    grouped = defaultdict(list)
    for s, i in zip(scores, idxs):
        row = chunks_df.loc[int(i)]
        grouped[row["paper_id"]].append({
            "score":    float(s),
            "chunk_id": int(row["chunk_id"]),
            "snippet":  row["chunk_text"][:500],
        })

    results = []
    for pid, chunks in grouped.items():
        chunks_sorted = sorted(chunks, key=lambda x: -x["score"])
        top_chunks    = chunks_sorted[:chunks_per_paper]
        row0          = chunks_df[chunks_df["paper_id"] == pid].iloc[0]
        results.append({
            "paper_id":   pid,
            "title":      row0["title"],
            "url":        row0["url"],
            "best_score": top_chunks[0]["score"],
            "abstract":   str(row0["abstract"])[:400],
            "chunks":     top_chunks,
        })

    results = sorted(results, key=lambda x: -x["best_score"])[:final_papers]

    # 4. Flat context list for the LLM prompt
    contexts = [c["snippet"] for r in results for c in r["chunks"]]

    return results, contexts
