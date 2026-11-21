"""Configuration profiles for baseline and upgraded (V2) ViCare RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmbeddingBackend = Literal["e5", "embeddinggemma2"]


@dataclass(frozen=True)
class ViCareConfig:
    """Configuration for a reproducible ViCare RAG experiment."""

    profile_name: str
    embedding_model: str
    embedding_backend: EmbeddingBackend
    chunk_size: int = 200
    overlap: int = 40
    n_chunks: int | None = None
    top_k: int = 5
    min_score: float = 0.86
    llm_model: str = "qwen2.5:3b"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_seed: int = 42
    chroma_path: str = "chroma_db"
    # Hybrid retrieval (BM25 + Dense RRF)
    use_hybrid_bm25: bool = False
    rrf_k: int = 60
    dense_weight: float = 0.65
    bm25_weight: float = 0.35
    hnsw_ef_search: int = 100
    # EmbeddingGemma 2 specific controls
    query_prompt_name: str = "QuestionAnswering"
    document_prompt_name: str = "Document"
    truncate_dim: int | None = None
    text_only_encoders: bool = True
    batch_size: int = 64
    # Guardrail & generation settings
    enforce_citations: bool = True
    allow_citation_repair: bool = True
    use_upgraded_guardrails: bool = True
    max_new_tokens: int = 320
    dev_seed: int = 2026
    dev_size: int = 100
    n_judge: int = 10


BASELINE_CONFIG = ViCareConfig(
    profile_name="baseline_e5_small",
    embedding_model="intfloat/multilingual-e5-small",
    embedding_backend="e5",
    chunk_size=200,
    overlap=40,
    n_chunks=None,
    top_k=5,
    min_score=0.86,
    llm_model="qwen2.5:3b",
    use_hybrid_bm25=False,
    hnsw_ef_search=100,
    enforce_citations=False,
    allow_citation_repair=False,
    use_upgraded_guardrails=False,
    max_new_tokens=256,
)

EMBEDDINGGEMMA2_CONFIG = ViCareConfig(
    profile_name="improved_v2_embeddinggemma2_hybrid",
    embedding_model="google/embeddinggemma-2",
    embedding_backend="embeddinggemma2",
    chunk_size=200,
    overlap=40,
    n_chunks=None,
    top_k=5,
    # Cosine similarity scale differs from E5; calibrate on dev vs OOS probes before final runs.
    min_score=0.67,
    llm_model="qwen2.5:3b",
    use_hybrid_bm25=True,
    rrf_k=60,
    dense_weight=0.65,
    bm25_weight=0.35,
    hnsw_ef_search=200,
    query_prompt_name="QuestionAnswering",
    document_prompt_name="Document",
    truncate_dim=None,
    text_only_encoders=True,
    batch_size=32,
    enforce_citations=True,
    allow_citation_repair=True,
    use_upgraded_guardrails=True,
    max_new_tokens=320,
)
