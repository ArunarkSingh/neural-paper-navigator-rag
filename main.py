#!/usr/bin/env python3
"""
main.py — CLI entrypoint for Paper Navigator.

Usage
-----
# Build the index from scratch (downloads dataset, embeds, saves to ./output):
    python main.py build

# Search interactively (loads pre-built index from ./output):
    python main.py search

# Search with a one-shot query:
    python main.py search --query "diffusion transformer text-to-image"

# Run latency benchmark:
    python main.py bench

Options
-------
  --out-dir     Directory for index / embeddings (default: output)
  --max-papers  Max papers to index (default: 15000)
  --no-rerank   Disable cross-encoder reranking
  --query       One-shot query string for 'search' mode
"""

import argparse
import os
import sys

from config import Config
from data_loader import load_arxiv_dataset, filter_dataset
from chunker import build_chunks_dataframe
from embedder import load_embedder, embed_texts, load_embeddings
from indexer import build_and_save, load_all
from search import load_reranker, search_papers, similar_papers, benchmark, pretty_print


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def cmd_build(cfg: Config) -> None:
    print("\n=== PAPER NAVIGATOR: BUILD ===\n")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1. Load & filter dataset
    ds, ds_name, col_map = load_arxiv_dataset(cfg.dataset_name)
    ds_filtered = filter_dataset(ds, col_map, cfg)

    # 2. Sample & chunk
    n = min(cfg.max_papers, len(ds_filtered))
    ds_small = ds_filtered.shuffle(seed=42).select(range(n))
    chunks_df = build_chunks_dataframe(ds_small, col_map, cfg)

    # 3. Embed
    embedder  = load_embedder(cfg)
    emb_path  = os.path.join(cfg.out_dir, "chunk_embeddings.f32.npy")
    mm, dim   = embed_texts(chunks_df["text"].tolist(), embedder, emb_path, cfg.embed_batch_size)

    # 4. Build & save FAISS index + chunk metadata
    build_and_save(mm, chunks_df, cfg)

    print("\nBuild complete ✅")
    print(f"Output directory: {cfg.out_dir}")


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------

def cmd_search(cfg: Config, query: str = None) -> None:
    print("\n=== PAPER NAVIGATOR: SEARCH ===\n")

    index, chunks_df = load_all(cfg)
    embedder         = load_embedder(cfg)
    reranker         = load_reranker(cfg)

    if query:
        # One-shot mode
        results = search_papers(query, embedder, index, chunks_df, cfg, reranker=reranker)
        pretty_print(results)

        if results:
            print("\n--- Similar papers to top result ---")
            sims = similar_papers(results[0], index, chunks_df)
            for s in sims:
                print(f"  {s['score']:.4f} | {s['title'][:100]} | {s['url']}")
    else:
        # Interactive REPL
        print("Type a query and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                q = input("Query > ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q.lower() in {"quit", "exit", "q"}:
                break
            if not q:
                continue
            results = search_papers(q, embedder, index, chunks_df, cfg, reranker=reranker)
            pretty_print(results)


# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------

def cmd_bench(cfg: Config) -> None:
    print("\n=== PAPER NAVIGATOR: BENCHMARK ===\n")

    index, chunks_df = load_all(cfg)
    embedder         = load_embedder(cfg)
    reranker         = load_reranker(cfg)

    bench_queries = [
        "vision transformer patch size ablation",
        "retrieval augmented generation evaluation metrics",
        "contrastive learning for sentence embeddings",
        "diffusion models for image generation",
        "robotics imitation learning behavior cloning",
    ]

    print("Latency (no rerank):", benchmark(bench_queries, embedder, index, chunks_df, cfg, reranker=None, runs=25))
    print("Latency (rerank)   :", benchmark(bench_queries, embedder, index, chunks_df, cfg, reranker=reranker, runs=10))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper Navigator — semantic search over ML ArXiv papers"
    )
    parser.add_argument(
        "command", choices=["build", "search", "bench"],
        help="build: create index | search: query the index | bench: latency test"
    )
    parser.add_argument("--out-dir",    default="output",  help="Index output directory")
    parser.add_argument("--max-papers", type=int, default=15000)
    parser.add_argument("--no-rerank",  action="store_true", help="Disable reranker")
    parser.add_argument("--query",      default=None,       help="One-shot query for search mode")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        out_dir      = args.out_dir,
        max_papers   = args.max_papers,
        use_reranker = not args.no_rerank,
    )

    if args.command == "build":
        cmd_build(cfg)
    elif args.command == "search":
        cmd_search(cfg, query=args.query)
    elif args.command == "bench":
        cmd_bench(cfg)
