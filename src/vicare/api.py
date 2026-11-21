"""FastAPI REST service exposing ViCare retrieval and grounded Q&A endpoints."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vicare.chunking import build_chunks
from vicare.config import BASELINE_CONFIG, EMBEDDINGGEMMA2_CONFIG, ViCareConfig
from vicare.data import load_vihealthqa
from vicare.embeddings import ViCareEmbedder
from vicare.rag import ViCareRAG
from vicare.retriever import ViCareRetriever


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Vietnamese health question")
    k: int = Field(default=5, ge=1, le=20, description="Number of unique source documents")
    min_score: float | None = Field(
        default=None, description="Optional override for similarity refusal threshold"
    )


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    index: int
    doc_id: str
    url: str
    score: float
    cited: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    refused: bool
    refusal_reason: str | None
    emergency_triggered: bool
    emergency_categories: list[str]
    citation_audit: dict[str, Any] | None
    latency_ms: float


def _build_default_rag() -> ViCareRAG:
    profile = os.environ.get("VICARE_PROFILE", "embeddinggemma2").strip().lower()
    config: ViCareConfig = (
        BASELINE_CONFIG if profile == "baseline" else EMBEDDINGGEMMA2_CONFIG
    )
    corpus, _, _ = load_vihealthqa(dev_seed=config.dev_seed, dev_size=config.dev_size)
    chunks = build_chunks(corpus, chunk_size=config.chunk_size, overlap=config.overlap)
    embedder = ViCareEmbedder(config)
    retriever = ViCareRetriever(config, embedder, chunks)
    return ViCareRAG(config, retriever)


def create_app(rag_engine: ViCareRAG | None = None) -> FastAPI:
    """Create a FastAPI app with an optional pre-injected ViCareRAG instance."""
    app = FastAPI(
        title="ViCare RAG API",
        version="0.1.0",
        description=(
            "Academic Vietnamese Health Q&A RAG API (AI4LIFE x Rikkei AI Challenge 24h). "
            "Not a medical diagnosis or clinical decision tool."
        ),
    )
    state: dict[str, ViCareRAG | None] = {"rag": rag_engine}

    def get_rag() -> ViCareRAG:
        if state["rag"] is None:
            state["rag"] = _build_default_rag()
        return state["rag"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "initialized": state["rag"] is not None}

    @app.get("/meta")
    def meta() -> dict[str, Any]:
        rag = get_rag()
        return {"config": asdict(rag.config)}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict[str, Any]:
        rag = get_rag()
        hits = rag.retriever.retrieve(req.question, k=req.k)
        return {"question": req.question, "hits": hits}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict[str, Any]:
        rag = get_rag()
        try:
            result = rag.ask(req.question, k=req.k, min_score=req.min_score)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG backend error: {exc}") from exc
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "refused": result["refused"],
            "refusal_reason": result["refusal_reason"],
            "emergency_triggered": result["emergency_triggered"],
            "emergency_categories": result["emergency_categories"],
            "citation_audit": result["citation_audit"],
            "latency_ms": result["latency_ms"],
        }

    return app


app = create_app()
