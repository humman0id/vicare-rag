"""ChromaDB vector indexer and deduplicated document retriever with telemetry."""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Any

import chromadb

from vicare.config import ViCareConfig
from vicare.embeddings import ViCareEmbedder


def compute_index_signature(config: ViCareConfig, data: list[dict[str, Any]]) -> str:
    """Deterministic signature tying a Chroma collection to data + embedding config."""
    hasher = hashlib.sha256()
    header = (
        f"{config.embedding_model}|{config.embedding_backend}|"
        f"{config.document_prompt_name}|{config.truncate_dim}|"
        f"{config.chunk_size}|{config.overlap}|l2|"
    )
    hasher.update(header.encode("utf-8"))
    for item in data:
        hasher.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


class ViCareRetriever:
    """Manages ChromaDB collection creation, indexing, and top-k document retrieval."""

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
            configuration={"hnsw": {"space": "l2"}},
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

    def retrieve(self, question: str, k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve top-k unique documents by cosine similarity (converted from L2)."""
        k = self.config.top_k if k is None else k
        if not isinstance(question, str) or not question.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        total = self.collection.count()
        if total == 0:
            return []

        n_candidates = min(total, max(k * 10, 30))
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(question.strip())],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            doc_id = meta["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            hits.append(
                {
                    "doc_id": doc_id,
                    "url": meta.get("url", ""),
                    "text": doc,
                    "score": 1.0 - float(dist) / 2.0,
                }
            )
            if len(hits) >= k:
                break
        return hits

    def telemetry(self) -> dict[str, Any]:
        """Return indexing statistics for transparent trade-off comparison."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "unique_docs": len(self.indexed_doc_ids),
            "newly_indexed_chunks": self.newly_indexed,
            "index_time_s": self.index_time_s,
        }
