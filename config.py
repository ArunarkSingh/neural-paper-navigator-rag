from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Output directory for index, embeddings, and metadata
    out_dir: str = "output"

    # Dataset
    dataset_name: str = "CShorten/ML-ArXiv-Papers"

    # Scale controls
    max_papers: int = 15000
    max_chunks_per_paper: int = 12

    # Chunking
    chunk_words: int = 220
    chunk_overlap: int = 40

    # Embedding model
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 64

    # Reranker
    use_reranker: bool = True
    rerank_model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1"

    # FAISS HNSW parameters
    hnsw_m: int = 32
    ef_search: int = 64
    ef_construction: int = 200

    # Search behaviour
    retrieve_k: int = 100
    final_papers: int = 10

    # ML keyword filter — keeps results modern and on-topic
    ml_hints: List[str] = field(default_factory=lambda: [
        "transformer", "vision transformer", "diffusion model", "denoising",
        "generative", "latent", "attention", "self-attention", "language model",
        "bert", "gpt", "neural network", "deep learning", "representation learning",
        "contrastive", "self-supervised", "reinforcement learning", "stable diffusion",
        "score-based", "denoising diffusion", "dit", "image synthesis", "text-to-image",
        "diffusion transformer", "generative ai", "large language model", "llm",
        "latent diffusion", "ddpm", "generation",
    ])

    # Terms that indicate classical math / PDE papers — excluded
    exclude_hints: List[str] = field(default_factory=lambda: [
        "pde", "partial differential equation", "anisotropic diffusion",
        "linear filtering", "numerical solution", "osmosis", "histogram prescription",
    ])
