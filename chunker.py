"""
chunker.py — converts raw paper records into overlapping word-level chunks
and returns a flat pandas DataFrame (one row per chunk).
"""

import re
from typing import List, Optional
import pandas as pd
from tqdm.auto import tqdm

from config import Config


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def make_pseudo_title(article: str, abstract: str) -> str:
    """Derive a title from the first meaningful line of the article or abstract."""
    if article:
        for line in article.split("\n"):
            line = line.strip()
            if len(line) >= 8:
                return line[:160]
    return (abstract[:160] + "...") if len(abstract) > 160 else abstract


def chunk_by_words(text: str, chunk_words: int, overlap: int) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_words - overlap)
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_words]
        if len(chunk) < max(40, chunk_words // 3):
            break
        chunks.append(" ".join(chunk))
    return chunks


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def build_chunks_dataframe(ds_small, col_map: dict, cfg: Config) -> pd.DataFrame:
    """
    Iterate over papers and produce a flat DataFrame of chunks.

    Each row contains:
        paper_id, title, abstract, url, chunk_id, chunk_text, text (for embedding)
    """
    abs_col     = col_map["abs_col"]
    article_col = col_map["article_col"]
    title_col   = col_map["title_col"]
    id_col      = col_map["id_col"]
    url_col     = col_map["url_col"]

    chunk_rows = []
    paper_counter = 0

    for ex in tqdm(ds_small, desc="Chunking papers"):
        abstract      = clean_text(str(ex.get(abs_col, "") or "")) if abs_col else ""
        article       = str(ex.get(article_col, "") or "") if article_col else ""
        article_clean = clean_text(article)

        # Paper ID
        pid = str(ex.get(id_col, "")).strip() if id_col else ""
        if not pid:
            pid = f"paper_{paper_counter}"
        paper_counter += 1

        # URL — construct ArXiv link if ID looks like one
        url = str(ex.get(url_col, "")).strip() if url_col else ""
        if not url and pid and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", pid):
            url = f"https://arxiv.org/abs/{pid}"

        # Title
        if title_col:
            title = clean_text(str(ex.get(title_col, "") or ""))
            if not title:
                title = make_pseudo_title(article, abstract)
        else:
            title = make_pseudo_title(article, abstract)

        # Choose text to chunk (prefer full article if long enough)
        base_text = article_clean if len(article_clean) > 600 else abstract
        if len(base_text) < 200:
            continue

        chunks = chunk_by_words(base_text, cfg.chunk_words, cfg.chunk_overlap)
        chunks = chunks[: cfg.max_chunks_per_paper]

        for j, ch in enumerate(chunks):
            # Prepend title + abstract context so each chunk retrieves better
            text_for_embed = f"Title: {title}\nAbstract: {abstract}\nChunk: {ch}"
            chunk_rows.append({
                "paper_id":   pid,
                "title":      title,
                "abstract":   abstract,
                "url":        url,
                "chunk_id":   j,
                "chunk_text": ch,
                "text":       text_for_embed,
            })

    df = pd.DataFrame(chunk_rows).reset_index(drop=True)
    print(f"Total chunks: {len(df)} | Unique papers: {df['paper_id'].nunique()}")
    return df
