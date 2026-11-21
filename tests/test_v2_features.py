"""Unit tests for V2 upgrades: Hybrid RRF, Vietnamese citation normalization, MRL ablation, and UI footer."""

import numpy as np

from vicare.evaluation import evaluate_matryoshka_ablation, format_matryoshka_table
from vicare.guardrails import verify_and_clean_citations
from vicare.retriever import fuse_rrf_candidates, tokenize_vi
from vicare.ui import format_status_footer


def test_vietnamese_prose_citation_normalization():
    raw = (
        "Đưa trẻ đi khám sớm (trích dẫn [1] và [2]). "
        "Cần theo dõi nhiệt độ (tài liệu 3) và không tự dùng thuốc [nguồn 4] [99]."
    )
    audit = verify_and_clean_citations(raw, num_sources=5)
    assert audit.valid_indices == [1, 2, 3, 4]
    assert audit.invalid_indices == [99]
    assert "(trích dẫn" not in audit.cleaned_text
    assert "(tài liệu" not in audit.cleaned_text
    assert "[1][2]" in audit.cleaned_text
    assert "[3]" in audit.cleaned_text
    assert "[4]" in audit.cleaned_text
    assert "[99]" not in audit.cleaned_text


def test_tokenize_vi_preserves_vietnamese_words():
    tokens = tokenize_vi("Đang chích ngừa viêm gan B, Covid-19?")
    assert "đang" in tokens
    assert "viêm" in tokens
    assert "covid" in tokens
    assert "19" in tokens


def test_fuse_rrf_candidates_promotes_consensus_and_preserves_dense_score():
    dense = [
        {"doc_id": "d1", "chunk_id": "d1#0", "url": "", "text": "A", "score": 0.80},
        {"doc_id": "d2", "chunk_id": "d2#0", "url": "", "text": "B", "score": 0.79},
    ]
    bm25 = [
        {"doc_id": "d2", "chunk_id": "d2#0", "url": "", "text": "B", "bm25_score": 12.5},
        {"doc_id": "d3", "chunk_id": "d3#0", "url": "", "text": "C", "bm25_score": 8.0},
    ]
    fused = fuse_rrf_candidates(dense, bm25, k=3, rrf_k=60, dense_weight=0.65, bm25_weight=0.35)
    # d2 appears in both dense (#2) and BM25 (#1), so RRF ranks it #1 while preserving dense score 0.79
    assert fused[0]["doc_id"] == "d2"
    assert fused[0]["score"] == 0.79
    assert fused[0]["bm25_score"] == 12.5
    assert len(fused) == 3


def test_matryoshka_ablation_slices_and_normalizes():
    rng = np.random.default_rng(42)
    doc_embs = rng.normal(size=(20, 768)).astype(np.float32)
    doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True)
    doc_ids = [f"doc-{i}" for i in range(20)]

    # Use doc 0 and doc 5 as exact queries -> Recall@1 should be 1.0 across dims
    q_embs = np.vstack([doc_embs[0], doc_embs[5]])
    questions = [
        {"id": "q0", "gold_doc_ids": ["doc-0"]},
        {"id": "q1", "gold_doc_ids": ["doc-5"]},
    ]
    rows = evaluate_matryoshka_ablation(
        doc_embs, doc_ids, q_embs, questions, dims=(768, 256, 128), k_values=(1, 3, 5, 10)
    )
    assert [r["dim"] for r in rows] == [768, 256, 128]
    assert rows[0]["recall@1"] == 1.0
    assert rows[1]["compression"] == "3.0x"
    table = format_matryoshka_table(rows)
    assert "MATRYOSHKA" in table
    assert "256" in table


def test_ui_status_footer_shows_refusal_reason_and_citations():
    refused_footer = format_status_footer(
        {"refused": True, "refusal_reason": "below_min_score", "top_score": 0.59, "min_score": 0.67}
    )
    assert "0.590" in refused_footer
    assert "ngoài phạm vi" in refused_footer

    ok_footer = format_status_footer(
        {
            "refused": False,
            "top_score": 0.79,
            "min_score": 0.67,
            "citation_audit": {"valid_indices": [1, 3]},
        }
    )
    assert "[1], [3]" in ok_footer
