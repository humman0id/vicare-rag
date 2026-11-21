"""Unit tests for chunking and PII masking."""

import pytest

from vicare.chunking import build_chunks, chunk_text
from vicare.data import mask_pii


def test_chunk_text_matches_baseline_assertion():
    assert chunk_text("a b c d e f g", 4, 1) == ["a b c d", "d e f g"]
    assert chunk_text("", 4, 1) == []


def test_chunk_text_validation():
    with pytest.raises(ValueError):
        chunk_text("hello world", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("hello world", 4, 4)
    with pytest.raises(ValueError):
        chunk_text("hello world", 4, -1)


def test_build_chunks_preserves_metadata():
    corpus = [{"doc_id": "train-1", "url": "https://example.com", "text": "one two three four five"}]
    chunks = build_chunks(corpus, chunk_size=3, overlap=1)
    assert len(chunks) == 2
    assert chunks[0]["id"] == "train-1#0"
    assert chunks[1]["id"] == "train-1#1"
    assert chunks[0]["doc_id"] == "train-1"
    assert chunks[1]["url"] == "https://example.com"


def test_mask_pii_redacts_vn_phone_and_email():
    raw = "Liên hệ 0912345678 hoặc +84987654321 và email bs.an@benhvien.vn nhé."
    masked = mask_pii(raw)
    assert "0912345678" not in masked
    assert "+84987654321" not in masked
    assert "bs.an@benhvien.vn" not in masked
    assert masked.count("[đã ẩn]") == 3
