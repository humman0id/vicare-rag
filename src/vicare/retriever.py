"""ChromaDB vector indexer and hybrid (Dense + BM25 RRF) deduplicated retriever with telemetry."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from vicare.config import ViCareConfig
from vicare.embeddings import ViCareEmbedder

_VI_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese text into lowercase Unicode word tokens for BM25."""
    return _VI_TOKEN_RE.findall(str(text or "").casefold())


def compute_index_signature(config: ViCareConfig, data: list[dict[str, Any]]) -> str:
    """Deterministic signature tying a Chroma collection to data + embedding config."""
    hasher = hashlib.sha256()
    header = (
        f"{config.embedding_model}|{config.embedding_backend}|"
        f"{config.document_prompt_name}|{config.truncate_dim}|"
        f"{config.chunk_size}|{config.overlap}|l2|ef{config.hnsw_ef_search}|"
    )
    hasher.update(header.encode("utf-8"))
    for item in data:
        hasher.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def fuse_rrf_candidates(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int,
    rrf_k: int = 60,
    dense_weight: float = 0.65,
    bm25_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """Fuse dense and BM25 per-document hit lists using weighted Reciprocal Rank Fusion.

    Preserves each document's true dense cosine similarity in `score` so calibrated
    similarity refusal thresholds (`MIN_SCORE`) remain strictly meaningful.
    """
    merged: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, 1):
        doc_id = hit["doc_id"]
        entry = merged.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "chunk_id": hit.get("chunk_id", ""),
                "url": hit.get("url", ""),
                "text": hit["text"],
                "score": float(hit["score"]),
                "dense_score": float(hit["score"]),
                "bm25_score": 0.0,
                "rrf_score": 0.0,
            },
        )
        entry["rrf_score"] += dense_weight / (rrf_k + rank)

    for rank, hit in enumerate(bm25_hits, 1):
        doc_id = hit["doc_id"]
        if doc_id not in merged:
            merged[doc_id] = {
                "doc_id": doc_id,
                "chunk_id": hit.get("chunk_id", ""),
                "url": hit.get("url", ""),
                "text": hit["text"],
                "score": float(hit.get("score", 0.0)),
                "dense_score": float(hit.get("score", 0.0)),
                "bm25_score": float(hit.get("bm25_score", 0.0)),
                "rrf_score": 0.0,
            }
        else:
            merged[doc_id]["bm25_score"] = max(
                merged[doc_id]["bm25_score"], float(hit.get("bm25_score", 0.0))
            )
        merged[doc_id]["rrf_score"] += bm25_weight / (rrf_k + rank)

    ranked = sorted(
        merged.values(),
        key=lambda item: (item["rrf_score"], item["dense_score"]),
        reverse=True,
    )
    return ranked[:k]


class ViCareRetriever:
    """Manages ChromaDB collection, optional BM25 lexical index, and top-k retrieval."""

    def __init__(
        self,
        config: ViCareConfig,
        embedder: ViCareEmbedder,
        chunks: list[dict[str, Any]],
        index_batch_size: int = 256,
    ) -> None:
        self.config = config
        self.embedder = embedder

        if config.n_chunks is not None and (
            not isinstance(config.n_chunks, int) or config.n_chunks <= 0
        ):
            raise ValueError("n_chunks must be None or a positive integer.")

        self.data = (
            chunks
            if config.n_chunks is None
            else random.Random(config.dev_seed).sample(
                chunks, k=min(config.n_chunks, len(chunks))
            )
        )
        if not self.data:
            raise ValueError("No chunks provided for indexing.")

        self.signature = compute_index_signature(config, self.data)
        self.collection_name = f"vicare_{self.signature}"
        self.client = chromadb.PersistentClient(path=config.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={
                "hnsw": {
                    "space": "l2",
                    "ef_search": config.hnsw_ef_search,
                    "ef_construction": max(100, config.hnsw_ef_search),
                }
            },
        )

        existing_ids = set(self.collection.get(include=[])["ids"])
        pending = [item for item in self.data if item["id"] not in existing_ids]

        t0 = time.perf_counter()
        for offset in range(0, len(pending), index_batch_size):
            batch = pending[offset : offset + index_batch_size]
            self.collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=self.embedder.embed_passages(
                    [item["text"] for item in batch], show_progress_bar=False
                ),
                metadatas=[
                    {"doc_id": item["doc_id"], "url": item.get("url", "")}
                    for item in batch
                ],
            )
        self.index_time_s = round(time.perf_counter() - t0, 3)
        self.newly_indexed = len(pending)
        self.indexed_doc_ids = {item["doc_id"] for item in self.data}

        if self.collection.count() != len(self.data):
            raise RuntimeError(
                f"Collection count mismatch: {self.collection.count()} != {len(self.data)}"
            )

        # Build lightweight in-memory BM25 index when hybrid retrieval is enabled.
        self.bm25: BM25Okapi | None = None
        self.bm25_index_time_s = 0.0
        if config.use_hybrid_bm25:
            tb = time.perf_counter()
            tokenized_corpus = [tokenize_vi(item["text"]) for item in self.data]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_index_time_s = round(time.perf_counter() - tb, 3)

    def _retrieve_dense(
        self, query_emb: list[float], max_docs: int
    ) -> list[dict[str, Any]]:
        total = self.collection.count()
        if total == 0:
            return []
        n_candidates = min(total, max(max_docs * 4, 50))
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            doc_id = meta["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            hits.append(
                {
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "url": meta.get("url", ""),
                    "text": doc,
                    "score": 1.0 - float(dist) / 2.0,
                }
            )
            if len(hits) >= max_docs:
                break
        return hits

    def _retrieve_bm25(self, question: str, max_docs: int) -> list[dict[str, Any]]:
        if self.bm25 is None:
            return []
        tokens = tokenize_vi(question)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        n_candidates = min(len(self.data), max(max_docs * 4, 50))
        if n_candidates <= 0:
            return []
        top_idx = np.argpartition(scores, -n_candidates)[-n_candidates:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx in top_idx:
            bm25_score = float(scores[idx])
            if bm25_score <= 0.0:
                break
            item = self.data[int(idx)]
            doc_id = item["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            hits.append(
                {
                    "chunk_id": item["id"],
                    "doc_id": doc_id,
                    "url": item.get("url", ""),
                    "text": item["text"],
                    "bm25_score": bm25_score,
                }
            )
            if len(hits) >= max_docs:
                break
        return hits

    def _hydrate_missing_dense_scores(
        self, query_emb: list[float], hits: list[dict[str, Any]]
    ) -> None:
        """Compute exact dense cosine similarity for BM25-only hits via Chroma lookup."""
        missing = [h for h in hits if h.get("dense_score", 0.0) == 0.0 and h.get("chunk_id")]
        if not missing:
            return
        ids = [h["chunk_id"] for h in missing]
        res = self.collection.get(ids=ids, include=["embeddings"])
        emb_map = dict(zip(res["ids"], res["embeddings"]))
        q_vec = np.asarray(query_emb, dtype=np.float32)
        for hit in missing:
            emb = emb_map.get(hit["chunk_id"])
            if emb is not None:
                sim = float(np.dot(q_vec, np.asarray(emb, dtype=np.float32)))
                hit["score"] = sim
                hit["dense_score"] = sim

    def retrieve(self, question: str, k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve top-k unique documents (dense-only or hybrid BM25 + dense RRF)."""
        k = self.config.top_k if k is None else k
        if not isinstance(question, str) or not question.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        total = self.collection.count()
        if total == 0:
            return []

        q_clean = question.strip()
        query_emb = self.embedder.embed_query(q_clean)

        if not self.config.use_hybrid_bm25 or self.bm25 is None:
            return self._retrieve_dense(query_emb, max_docs=k)

        pool_size = max(k * 5, 30)
        dense_hits = self._retrieve_dense(query_emb, max_docs=pool_size)
        bm25_hits = self._retrieve_bm25(q_clean, max_docs=pool_size)
        fused = fuse_rrf_candidates(
            dense_hits,
            bm25_hits,
            k=k,
            rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            bm25_weight=self.config.bm25_weight,
        )
        self._hydrate_missing_dense_scores(query_emb, fused)
        return fused

    def telemetry(self) -> dict[str, Any]:
        """Return indexing statistics for transparent trade-off comparison."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "unique_docs": len(self.indexed_doc_ids),
            "newly_indexed_chunks": self.newly_indexed,
            "index_time_s": self.index_time_s,
            "hybrid_bm25": self.config.use_hybrid_bm25,
            "bm25_index_time_s": self.bm25_index_time_s,
        }
