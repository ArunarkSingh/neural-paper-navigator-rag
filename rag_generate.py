"""
rag_generate.py — Answer generation for the RAG pipeline.

Uses Groq (Llama 3.1-8B) to generate grounded answers from retrieved contexts.
Answers are constrained to only use information from the retrieved passages.
"""

import time
from typing import List, Dict

from groq import Groq


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant that answers questions about machine learning papers.
You are given retrieved context passages from ML papers.
Answer the question concisely and accurately using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Do NOT add knowledge from outside the context.
"""


def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    contexts: List[str],
    groq_client: Groq,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    """
    Generate a grounded answer using Groq Llama 3.1-8B.

    Temperature is kept low (0.1) to maximise faithfulness to context.
    """
    prompt = build_prompt(question, contexts)
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full RAG query (retrieval + generation, timed)
# ---------------------------------------------------------------------------

def rag_query(
    question: str,
    embedder,
    index,
    bm25,
    chunks_df,
    cfg,
    groq_client: Groq,
    reranker=None,
    retrieval_mode: str = "hybrid",
    final_papers: int = 5,
    model: str = "llama-3.1-8b-instant",
    verbose: bool = True,
) -> Dict:
    """
    End-to-end RAG pipeline: retrieve → (rerank) → generate.

    Returns a dict containing the question, answer, contexts,
    source papers, retrieval mode, and timing info.
    """
    from rag_retrieval import search_papers_rag

    t0 = time.time()
    results, contexts = search_papers_rag(
        question,
        embedder=embedder,
        index=index,
        bm25=bm25,
        chunks_df=chunks_df,
        cfg=cfg,
        reranker=reranker,
        retrieval_mode=retrieval_mode,
        final_papers=final_papers,
    )
    retrieval_ms = (time.time() - t0) * 1000

    t1 = time.time()
    answer = generate_answer(question, contexts, groq_client, model=model)
    generation_ms = (time.time() - t1) * 1000

    output = {
        "question":      question,
        "answer":        answer,
        "contexts":      contexts,
        "source_papers": [{"title": r["title"], "url": r["url"]} for r in results],
        "retrieval_mode": retrieval_mode,
        "retrieval_ms":  round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
    }

    if verbose:
        print(f"\n[{retrieval_mode.upper()} | rerank={reranker is not None}]")
        print(f"Retrieval: {retrieval_ms:.0f}ms | Generation: {generation_ms:.0f}ms")
        print(f"\nQ: {question}")
        print(f"\nA: {answer}")
        print("\nSources:")
        for p in output["source_papers"]:
            print(f"  • {p['title'][:90]}")

    return output
