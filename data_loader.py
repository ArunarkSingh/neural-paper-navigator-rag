"""
data_loader.py — loads an ArXiv-like HuggingFace dataset and applies
an ML keyword filter to remove off-topic / classical-math papers.
"""

import re
from typing import Optional, Tuple
from datasets import load_dataset, Dataset
from config import Config


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def pick_col(cols: list, options: list) -> Optional[str]:
    """Return the first option that exists in cols, else None."""
    for o in options:
        if o in cols:
            return o
    return None


def detect_columns(cols: list) -> dict:
    return {
        "title_col":   pick_col(cols, ["title", "paper_title", "article_title"]),
        "abs_col":     pick_col(cols, ["abstract", "summary", "paper_abstract"]),
        "article_col": pick_col(cols, ["article", "full_text", "text"]),
        "cat_col":     pick_col(cols, ["categories", "category", "primary_category", "subject"]),
        "id_col":      pick_col(cols, ["id", "arxiv_id", "paper_id", "article_id"]),
        "url_col":     pick_col(cols, ["url", "pdf_url", "arxiv_url", "link"]),
    }


# ---------------------------------------------------------------------------
# Dataset loading (tries multiple candidate names)
# ---------------------------------------------------------------------------

CANDIDATE_DATASETS = [
    "CShorten/ML-ArXiv-Papers",
    "ccdv/arxiv-summarization",
    "arxiv_dataset",
    "arxiv",
]


def load_arxiv_dataset(dataset_name: Optional[str] = None) -> Tuple[Dataset, str, dict]:
    """
    Load an ArXiv-like dataset from HuggingFace.

    Returns
    -------
    ds      : the train split as a Dataset
    name    : the dataset name that loaded successfully
    col_map : dict mapping logical roles to actual column names
    """
    candidates = [dataset_name] if dataset_name else CANDIDATE_DATASETS

    last_err = None
    for name in candidates:
        if name is None:
            continue
        try:
            ds_dict = load_dataset(name)
            split = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
            ds = ds_dict[split]
            print(f"Loaded: {name} | split: {split} | rows: {len(ds)} | cols: {ds.column_names}")
            col_map = detect_columns(ds.column_names)
            if col_map["abs_col"] is None and col_map["article_col"] is None:
                raise ValueError(f"No usable text column found. Columns: {ds.column_names}")
            return ds, name, col_map
        except Exception as e:
            last_err = e
            print(f"Failed: {name} | {str(e)[:140]}")

    raise RuntimeError(f"Could not load any dataset. Last error: {last_err}")


# ---------------------------------------------------------------------------
# ML relevance filter
# ---------------------------------------------------------------------------

def build_filter(col_map: dict, cfg: Config):
    """Return a filter function compatible with Dataset.filter()."""
    abs_col     = col_map["abs_col"]
    article_col = col_map["article_col"]
    cat_col     = col_map["cat_col"]

    target_cats = {"cs.LG", "cs.CV", "cs.AI", "stat.ML"}

    def has_target_cat(ex) -> bool:
        if not cat_col:
            return False
        c = str(ex.get(cat_col, "") or "")
        return any(t in c for t in target_cats)

    def looks_ml(ex) -> bool:
        a   = str(ex.get(abs_col, "") or "") if abs_col else ""
        art = str(ex.get(article_col, "") or "") if article_col else ""
        text = (a + " " + art[:4000]).lower()
        has_modern  = any(h in text for h in cfg.ml_hints)
        has_classic = any(e in text for e in cfg.exclude_hints)
        return has_modern and not has_classic

    return has_target_cat, looks_ml


def filter_dataset(ds: Dataset, col_map: dict, cfg: Config) -> Dataset:
    """Apply category or keyword filter; fall back to unfiltered if too small."""
    has_target_cat, looks_ml = build_filter(col_map, cfg)

    if col_map["cat_col"]:
        ds_f = ds.filter(has_target_cat)
        print(f"After category filter: {len(ds_f)}")
        if len(ds_f) < 2000:
            ds_f = ds.filter(looks_ml)
            print(f"Category filter too small → keyword filter: {len(ds_f)}")
    else:
        ds_f = ds.filter(looks_ml)
        print(f"After keyword filter: {len(ds_f)}")

    if len(ds_f) < 2000:
        print("Filter produced a small set. Using unfiltered dataset.")
        ds_f = ds

    return ds_f
