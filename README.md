# Neural Research Paper Navigator: Hybrid Retrieval and RAG for ML Literature

Semantic search and retrieval-augmented Q&A over ~15,000 ML ArXiv papers.

Two-stage retrieval (dense ANN + cross-encoder reranking) extended with a hybrid BM25+dense pipeline, Groq-powered answer generation, and a custom LLM-as-judge evaluation framework benchmarking retrieval quality.

---

## Architecture

```
Dataset (117k papers)
        │
        ▼
  ML keyword filter  ──►  81,272 papers
        │
        ▼
  Chunk (220 words, 40 overlap)  ──►  15,910 chunks from 15,000 papers
        │
        ├──► FAISS HNSW index  (BAAI/bge-small-en-v1.5, dim=384)
        │
        └──► BM25Okapi index   (rank-bm25)
                │
                ▼
        ┌──────────────────────────────────────────┐
        │  Query                                   │
        │    ├─ Dense retrieve (FAISS)             │
        │    ├─ Sparse retrieve (BM25)             │
        │    └─ Fuse with RRF (k=60)               │
        │         │                                │
        │         ▼                                │
        │  Cross-encoder rerank                    │
        │  (mixedbread-ai/mxbai-rerank-xsmall-v1)  │
        │         │                                │
        │         ▼                                │
        │  Groq Llama-3.1-8B                       │
        │  (grounded answer generation)            │
        └──────────────────────────────────────────┘
```

### Key design decisions

**Chunking with context prepending** — each chunk is embedded as `"Title: {title}\nAbstract: {abstract}\nChunk: {text}"`, so the dense retriever sees the paper's topic even in a mid-paper chunk.

**Hybrid retrieval with RRF** — BM25 captures exact keyword matches (model names, acronyms) that dense embeddings miss; RRF fuses both ranked lists without score normalisation.

**Cross-encoder reranking** — after retrieving 100 candidates, a cross-encoder rescores query-chunk pairs for precision. This is the largest single latency cost but meaningfully improves result quality.

**Constrained generation** — the system prompt explicitly forbids the LLM from using knowledge outside the retrieved context, making faithfulness measurable.

---

## Evaluation Results

Evaluated on 5 ML queries using Groq Llama-3.1-8B as the LLM judge across three RAGAS-inspired metrics:

| Metric | Dense-only | Hybrid (BM25 + Dense) | Δ | Winner |
|---|---|---|---|---|
| **Faithfulness** | 0.833 | **0.917** | +0.083 | Hybrid ✓ |
| **Answer Relevancy** | **0.920** | 0.900 | −0.020 | Tie |
| **Context Precision** | 0.419 | **0.425** | +0.006 | Hybrid ✓ |

**Key finding:** Hybrid retrieval improves faithfulness by 8.3 points — answers are more grounded in the retrieved context. Answer relevancy is comparable between both modes. The gains come from BM25's ability to match exact technical terms (model names, method acronyms) that dense embeddings can under-weight.

### Latency

| Mode | p50 | p95 |
|---|---|---|
| Dense only | 170ms | 176ms |
| Dense + rerank | 323ms | 334ms |
| RAG (hybrid + rerank + generation) | ~420ms retrieval + ~200ms generation |

## Retrieval & System Analysis

### Dataset Filtering

The initial ArXiv dataset contains many classical mathematics and PDE papers where the word *diffusion* refers to differential equations rather than modern diffusion models.  
A keyword-based ML filter removes non-ML papers before indexing.

<p align="center">
  <img src="/plots/01_dataset_filter.png" width="70%">
</p>

This reduces the dataset from **117,592 papers → 81,272 ML-relevant papers**, improving retrieval precision.

---

### Chunk Distribution

Papers are chunked into overlapping windows of **220 words with 40-word overlap**.

<p align="center">
  <img src="/plots/02_chunks_per_paper.png" width="70%">
</p>

Most papers generate **one chunk** because the dataset contains titles + abstracts rather than full text.

---

### Retrieval Score Distributions

Dense retrieval and cross-encoder reranking produce very different score distributions.

<p align="center">
  <img src="/plots/03_score_distributions.png" width="70%">
</p>

Observations:

- Dense retrieval produces tightly clustered similarity scores (~0.70–0.82)
- Cross-encoder reranking produces **highly separable relevance scores**
- Reranking enables more precise ordering of top results

---

### Query Latency

Latency measured across repeated queries.

<p align="center">
  <img src="/plots/04_latency_violin.png" width="70%">
</p>

Cross-encoder reranking increases latency but significantly improves precision.

---

### RAG Retrieval Evaluation

Hybrid retrieval was evaluated against dense-only retrieval using a **RAGAS-style evaluation** with an LLM judge.

<p align="center">
  <img src="/plots/05_ragas_comparison.png" width="70%">
</p>

Hybrid retrieval improves **faithfulness and context precision** while maintaining similar answer relevancy.

---

## Project Structure

```
paper-navigator/
├── config.py           # Config dataclass — all hyperparameters in one place
├── data_loader.py      # Dataset loading and ML keyword filtering
├── chunker.py          # Overlapping word-level chunking with context prepending
├── embedder.py         # SentenceTransformer embedding + memmap storage
├── indexer.py          # FAISS HNSW index build, save, and load
├── search.py           # Dense search, cross-encoder reranking, similar_papers
├── visualizations.py   # Dataset/score/latency plots
├── main.py             # CLI: build | search | bench
│
├── rag_retrieval.py    # BM25 index, hybrid RRF retrieval, search_papers_rag
├── rag_generate.py     # Groq answer generation, full rag_query pipeline
├── rag_eval.py         # LLM-as-judge scoring (faithfulness, relevancy, precision)
├── rag_visualize.py    # Comparison bar chart, CSV/JSON result saving
└── rag_main.py         # CLI: ask | chat | eval
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the index

Downloads the dataset, filters to ML papers, chunks, embeds, and saves FAISS + BM25 indexes to `./output`.

```bash
python main.py build
```

This takes ~5–10 minutes on first run (embedding 15,910 chunks). Everything is cached to `./output` — subsequent runs load instantly.

### 3. Search (semantic search only)

```bash
# One-shot query
python main.py search --query "diffusion transformer DiT text-to-image"

# Interactive REPL
python main.py search

# Latency benchmark
python main.py bench
```

### 4. RAG — ask questions, get grounded answers

Set your Groq API key first (free at [console.groq.com](https://console.groq.com)):

```bash
export GROQ_API_KEY=gsk_...
```

```bash
# Single question (hybrid retrieval, default)
python rag_main.py ask "How do vision transformers use self-attention for image classification?"

# Single question with dense-only retrieval
python rag_main.py ask "How does contrastive learning work?" --mode dense

# Interactive Q&A loop
python rag_main.py chat

# Run full evaluation: dense vs hybrid on 5 queries + RAGAS scoring + plots
python rag_main.py eval
```

---

## Configuration

All hyperparameters live in `config.py`:

```python
@dataclass
class Config:
    max_papers:        int = 15000
    chunk_words:       int = 220
    chunk_overlap:     int = 40
    embed_model_name:  str = "BAAI/bge-small-en-v1.5"
    rerank_model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1"
    retrieve_k:        int = 100
    final_papers:      int = 10
```

---

## Dataset

**CShorten/ML-ArXiv-Papers** — 117,592 ArXiv paper titles and abstracts. After ML keyword filtering (transformer, diffusion, contrastive, LLM, etc.) and exclusion of classical math/PDE papers: **81,272 papers**. 15,000 sampled for indexing → **15,910 chunks**.

---

## Models

| Component | Model | Size |
|---|---|---|
| Dense embedder | BAAI/bge-small-en-v1.5 | 33M params |
| Cross-encoder reranker | mixedbread-ai/mxbai-rerank-xsmall-v1 | 70M params |
| Answer generation | Groq Llama-3.1-8B-Instant | 8B params (API) |
| Evaluation judge | Groq Llama-3.1-8B-Instant | 8B params (API) |

All local models run on CPU. No GPU required.

---

## Requirements

- Python 3.10+
- ~2GB disk for index + embeddings
- Groq API key (free tier) for RAG features
