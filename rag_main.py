#!/usr/bin/env python3
"""
rag_main.py — CLI entrypoint for the RAG extension of Paper Navigator.

Requires a pre-built index (run `python main.py build` first).

Usage
-----
# Ask a single question (hybrid retrieval, default):
    python rag_main.py ask "How do vision transformers use self-attention?"

# Ask with dense-only retrieval:
    python rag_main.py ask "How do vision transformers use self-attention?" --mode dense

# Interactive Q&A loop:
    python rag_main.py chat

# Run the full evaluation (dense vs hybrid on 5 queries):
    python rag_main.py eval

Options
-------
  --out-dir   Index directory (default: output)
  --mode      Retrieval mode: hybrid | dense  (default: hybrid)
  --no-rerank Disable cross-encoder reranking
  --papers    Number of papers to retrieve per query (default: 5)

Environment variables
---------------------
  GROQ_API_KEY   — required for answer generation and evaluation
"""

import argparse
import os
import sys
import time

from groq import Groq

from config import Config
from indexer import load_all
from embedder import load_embedder
from search import load_reranker
from rag_retrieval import build_bm25_index
from rag_generate import rag_query
from rag_eval import evaluate_outputs
from rag_visualize import (
    plot_ragas_comparison,
    print_comparison_table,
    print_per_query_comparison,
    save_results_json,
)


# ---------------------------------------------------------------------------
# Eval queries — corpus-appropriate ML topics
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    "What are the advantages of contrastive learning for self-supervised visual representations?",
    "How do vision transformers use self-attention for image classification?",
    "How do GANs enable image editing and style manipulation?",
    "What methods improve few-shot learning generalization?",
    "How does knowledge distillation compress large neural networks?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_everything(cfg: Config):
    """Load index, embedder, reranker, and build BM25 index."""
    print("Loading index and models...")
    index, chunks_df = load_all(cfg)
    embedder         = load_embedder(cfg)
    reranker         = load_reranker(cfg)
    bm25             = build_bm25_index(chunks_df)
    return index, chunks_df, embedder, reranker, bm25


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("  Get a free key at https://console.groq.com")
        print("  Then run:  export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ask(cfg: Config, question: str, mode: str) -> None:
    index, chunks_df, embedder, reranker, bm25 = load_everything(cfg)
    groq_client = get_groq_client()

    rag_query(
        question=question,
        embedder=embedder,
        index=index,
        bm25=bm25,
        chunks_df=chunks_df,
        cfg=cfg,
        groq_client=groq_client,
        reranker=reranker,
        retrieval_mode=mode,
        final_papers=cfg.final_papers,
        verbose=True,
    )


def cmd_chat(cfg: Config, mode: str) -> None:
    index, chunks_df, embedder, reranker, bm25 = load_everything(cfg)
    groq_client = get_groq_client()

    print(f"\n=== PAPER NAVIGATOR RAG — Interactive ({mode} retrieval) ===")
    print("Type a question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            q = input("Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue

        rag_query(
            question=q,
            embedder=embedder,
            index=index,
            bm25=bm25,
            chunks_df=chunks_df,
            cfg=cfg,
            groq_client=groq_client,
            reranker=reranker,
            retrieval_mode=mode,
            final_papers=cfg.final_papers,
            verbose=True,
        )
        print()


def cmd_eval(cfg: Config) -> None:
    print("\n=== PAPER NAVIGATOR RAG — Evaluation ===\n")

    index, chunks_df, embedder, reranker, bm25 = load_everything(cfg)
    groq_client = get_groq_client()

    # --- Generate answers for both retrieval modes ---
    print(f"Running RAG pipeline on {len(EVAL_QUERIES)} queries × 2 modes...\n")

    dense_outputs, hybrid_outputs = [], []
    for i, q in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{len(EVAL_QUERIES)}] {q[:70]}...")
        dense_outputs.append(rag_query(
            q, embedder, index, bm25, chunks_df, cfg, groq_client,
            reranker=reranker, retrieval_mode="dense",
            final_papers=cfg.final_papers, verbose=False,
        ))
        hybrid_outputs.append(rag_query(
            q, embedder, index, bm25, chunks_df, cfg, groq_client,
            reranker=reranker, retrieval_mode="hybrid",
            final_papers=cfg.final_papers, verbose=False,
        ))
        time.sleep(1.0)     # Groq free-tier rate limit

    print("\nDone generating answers.\n")

    # --- Score ---
    print("Scoring dense-only pipeline...")
    dense_result  = evaluate_outputs(dense_outputs,  groq_client, label="Dense")

    print("\nScoring hybrid pipeline...")
    hybrid_result = evaluate_outputs(hybrid_outputs, groq_client, label="Hybrid")

    # --- Report ---
    print_comparison_table(dense_result, hybrid_result, cfg.out_dir)
    print_per_query_comparison(dense_outputs, hybrid_outputs)

    plot_ragas_comparison(
        dense_result, hybrid_result,
        eval_queries=EVAL_QUERIES,
        out_dir=os.path.join(cfg.out_dir, "plots"),
        groq_model="llama-3.1-8b-instant",
    )

    save_results_json(
        EVAL_QUERIES, dense_result, hybrid_result,
        dense_outputs, hybrid_outputs,
        out_dir=cfg.out_dir,
        groq_model="llama-3.1-8b-instant",
        embed_model=cfg.embed_model_name,
        rerank_model=cfg.rerank_model_name,
    )

    print("\n✅ Evaluation complete!")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper Navigator RAG — retrieval-augmented Q&A over ML ArXiv papers"
    )
    parser.add_argument(
        "command", choices=["ask", "chat", "eval"],
        help="ask: one-shot Q&A | chat: interactive loop | eval: dense vs hybrid benchmark",
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="Question string (required for 'ask' command)")
    parser.add_argument("--out-dir",    default="output",   help="Index directory (default: output)")
    parser.add_argument("--mode",       default="hybrid",   choices=["hybrid", "dense"])
    parser.add_argument("--no-rerank",  action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--papers",     type=int, default=5, help="Papers to retrieve per query")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        out_dir      = args.out_dir,
        use_reranker = not args.no_rerank,
        final_papers = args.papers,
    )

    if args.command == "ask":
        if not args.question:
            print("ERROR: provide a question string after 'ask'")
            print("  Example: python rag_main.py ask \"How do ViTs use self-attention?\"")
            sys.exit(1)
        cmd_ask(cfg, args.question, args.mode)

    elif args.command == "chat":
        cmd_chat(cfg, args.mode)

    elif args.command == "eval":
        cmd_eval(cfg)
