"""Document chunking utilities for ViCare corpus."""

from __future__ import annotations

from typing import Any


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word chunks.

    Preserves exact baseline semantics (size > 0, 0 <= overlap < size).
    """
    if (
        not isinstance(size, int)
        or not isinstance(overlap, int)
        or size <= 0
        or not (0 <= overlap < size)
    ):
        raise ValueError(
            "Require integer size > 0 and 0 <= overlap < size."
        )
    words = str(text or "").split()
    if not words:
        return []

    parts: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return parts


def build_chunks(
    corpus: list[dict[str, Any]],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    """Expand corpus documents into chunk records retaining doc_id and url."""
    chunks: list[dict[str, Any]] = []
    for doc in corpus:
        for idx, part in enumerate(chunk_text(doc["text"], size=chunk_size, overlap=overlap)):
            chunks.append(
                {
                    "id": f"{doc['doc_id']}#{idx}",
                    "doc_id": doc["doc_id"],
                    "url": doc.get("url", ""),
                    "text": part,
                }
            )
    return chunks
