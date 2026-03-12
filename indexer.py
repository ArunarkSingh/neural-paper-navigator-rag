"""
indexer.py — builds and persists a FAISS HNSW index over normalised
chunk embeddings; also handles saving / loading the chunk metadata.
"""

import os
import numpy as np
import faiss
import pandas as pd

from config import Config


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, cfg: Config) -> faiss.IndexHNSWFlat:
    """
    Build a FAISS HNSW index using inner-product similarity on
    L2-normalised vectors (equivalent to cosine similarity).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = cfg.ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = cfg.ef_search
    print(f"FAISS index built: {index.ntotal} vectors | dim: {dim}")
    return index


def save_index(index: faiss.IndexHNSWFlat, path: str) -> None:
    faiss.write_index(index, path)
    print(f"Saved FAISS index: {path}")


def load_index(path: str, ef_search: int = 64) -> faiss.IndexHNSWFlat:
    index = faiss.read_index(path)
    index.hnsw.efSearch = ef_search
    print(f"Loaded FAISS index: {index.ntotal} vectors")
    return index


# ---------------------------------------------------------------------------
# Chunk metadata (parquet)
# ---------------------------------------------------------------------------

def save_chunks(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)
    print(f"Saved chunk metadata: {path}")


def load_chunks(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded chunk metadata: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Convenience: build + persist everything in one call
# ---------------------------------------------------------------------------

def build_and_save(embeddings: np.ndarray, chunks_df: pd.DataFrame, cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)

    index_path  = os.path.join(cfg.out_dir, "faiss_chunks_hnsw.index")
    chunks_path = os.path.join(cfg.out_dir, "chunks_meta.parquet")

    index = build_faiss_index(embeddings, cfg)
    save_index(index, index_path)
    save_chunks(chunks_df, chunks_path)

    return index


def load_all(cfg: Config):
    """Load a pre-built index and chunk metadata from disk."""
    index_path  = os.path.join(cfg.out_dir, "faiss_chunks_hnsw.index")
    chunks_path = os.path.join(cfg.out_dir, "chunks_meta.parquet")

    index     = load_index(index_path, cfg.ef_search)
    chunks_df = load_chunks(chunks_path)
    return index, chunks_df
