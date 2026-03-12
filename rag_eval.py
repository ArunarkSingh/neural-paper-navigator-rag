"""
rag_eval.py — LLM-as-judge evaluation for the RAG pipeline.

Scores three RAGAS-inspired metrics using Groq Llama 3.1-8B as the judge:

  - Faithfulness:       Are all claims in the answer supported by the context?
  - Answer Relevancy:   Does the answer directly address the question?
  - Context Precision:  Are the retrieved passages actually useful for the question?

Results are compared between dense-only and hybrid retrieval.
"""

import re
import json
import time
import numpy as np
from typing import List, Dict

from groq import Groq


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def groq_call(prompt: str, groq_client: Groq, model: str = "llama-3.1-8b-instant") -> str:
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# Are all factual claims in the answer supported by the retrieved context?
# ---------------------------------------------------------------------------

def score_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    groq_client: Groq,
    model: str = "llama-3.1-8b-instant",
) -> float:
    ctx = "\n\n".join(contexts)
    prompt = f"""You are evaluating whether an AI answer is faithful to the retrieved context.

CONTEXT:
{ctx}

ANSWER:
{answer}

Identify each factual claim in the answer. For each claim, check if it is directly supported by the context above.
Return ONLY valid JSON like: {{"supported": 3, "total": 4}}
where "supported" = number of claims supported by context, "total" = total claims."""

    try:
        raw = groq_call(prompt, groq_client, model)
        raw = re.search(r'\{.*?\}', raw, re.DOTALL).group()
        d = json.loads(raw)
        return d["supported"] / d["total"] if d["total"] > 0 else 1.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy
# Does the answer directly and completely address the question?
# ---------------------------------------------------------------------------

def score_answer_relevancy(
    question: str,
    answer: str,
    groq_client: Groq,
    model: str = "llama-3.1-8b-instant",
) -> float:
    prompt = f"""You are evaluating whether an AI answer is relevant to the question asked.

QUESTION: {question}

ANSWER: {answer}

Rate the answer relevancy on a scale from 0.0 to 1.0:
- 1.0 = directly and completely answers the question
- 0.5 = partially answers the question
- 0.0 = irrelevant or refuses to answer

Return ONLY valid JSON like: {{"score": 0.8}}"""

    try:
        raw = groq_call(prompt, groq_client, model)
        raw = re.search(r'\{.*?\}', raw, re.DOTALL).group()
        return float(json.loads(raw)["score"])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Metric 3: Context Precision
# Are the retrieved passages useful for answering the question?
# ---------------------------------------------------------------------------

def score_context_precision(
    question: str,
    contexts: List[str],
    groq_client: Groq,
    model: str = "llama-3.1-8b-instant",
) -> float:
    scores = []
    for ctx in contexts:
        prompt = f"""You are evaluating whether a retrieved context passage is useful for answering a question.

QUESTION: {question}

CONTEXT PASSAGE:
{ctx}

Is this context passage useful for answering the question?
Return ONLY valid JSON like: {{"useful": true}} or {{"useful": false}}"""

        try:
            raw = groq_call(prompt, groq_client, model)
            raw = re.search(r'\{.*?\}', raw, re.DOTALL).group()
            scores.append(1.0 if json.loads(raw)["useful"] else 0.0)
        except Exception:
            scores.append(float("nan"))

    valid = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Evaluate a full set of RAG outputs
# ---------------------------------------------------------------------------

def evaluate_outputs(
    outputs: List[Dict],
    groq_client: Groq,
    label: str = "",
    model: str = "llama-3.1-8b-instant",
    sleep_between: float = 0.5,
) -> Dict:
    """
    Score all outputs for faithfulness, answer relevancy, and context precision.
    Returns a dict with mean scores for each metric.
    """
    faith, relevancy, precision = [], [], []

    for i, o in enumerate(outputs):
        print(f"  [{i+1}/{len(outputs)}] {o['question'][:65]}...")
        faith.append(score_faithfulness(o["question"], o["answer"], o["contexts"], groq_client, model))
        relevancy.append(score_answer_relevancy(o["question"], o["answer"], groq_client, model))
        precision.append(score_context_precision(o["question"], o["contexts"], groq_client, model))
        time.sleep(sleep_between)   # stay within Groq free-tier rate limit

    result = {
        "faithfulness":      float(np.nanmean(faith)),
        "answer_relevancy":  float(np.nanmean(relevancy)),
        "context_precision": float(np.nanmean(precision)),
        "_raw": {
            "faithfulness":      faith,
            "answer_relevancy":  relevancy,
            "context_precision": precision,
        },
    }

    if label:
        print(f"\n{label} scores:")
        for k, v in result.items():
            if not k.startswith("_"):
                print(f"  {k:25s}: {v:.4f}")

    return result
