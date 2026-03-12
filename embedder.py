"""
embedder.py — encodes chunk texts into L2-normalised float32 embeddings
using a SentenceTransformer model, backed by a memory-mapped numpy array
so the full embedding matrix never has to fit in RAM at once.
"""

import os
import time
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from config import Config


def load_embedder(cfg: Config) -> SentenceTransformer:
    print(f"Loading embedding model: {cfg.embed_model_name}")
    return SentenceTransformer(cfg.embed_model_name)


def embed_texts(
    texts: List[str],
    embedder: SentenceTransformer,
    out_path: str,
    batch_size: int,
) -> Tuple[np.memmap, int]:
    """
    Encode `texts` in batches and write results to a memory-mapped file.

    Returns
    -------
    mm  : np.memmap of shape (N, dim) with float32 normalised embeddings
    dim : embedding dimensionality
    """
    # Infer dimension from a small sample
    sample = embedder.encode(texts[:2], normalize_embeddings=True)
    dim = sample.shape[1]

    mm = np.memmap(out_path, dtype="float32", mode="w+", shape=(len(texts), dim))

    t0 = time.time()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        embs  = embedder.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        mm[i : i + len(batch)] = embs

    mm.flush()
    elapsed = time.time() - t0
    print(f"Embeddings: {mm.shape} | dim: {dim} | time: {elapsed:.1f}s")
    return mm, dim


def load_embeddings(out_path: str, n: int, dim: int) -> np.memmap:
    """Re-open an existing embedding memmap (read-only)."""
    return np.memmap(out_path, dtype="float32", mode="r", shape=(n, dim))
