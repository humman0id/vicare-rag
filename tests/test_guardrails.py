"""Unit tests for emergency detection, citation verification, refusal calibration, and RAG flow."""

from vicare.config import EMBEDDINGGEMMA2_CONFIG
from vicare.guardrails import (
    CRISIS_ADDENDUM,
    EMERGENCY,
    REFUSAL,
    WARNING,
    calibrate_min_score,
    detect_emergency,
    format_answer,
    verify_and_clean_citations,
)
from vicare.rag import ViCareRAG


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, question: str, k: int = 5):
        return self._hits[:k]


def test_emergency_detection_diacritics_and_negation():
    # Diacritic-free input still detected
    is_em, cats = detect_emergency("Tre bi sot cao kem co giat thi lam gi?", upgraded=True)
    assert is_em
    assert "neurological_stroke" in cats

    # Negated symptom does not trigger false alarm
    is_em_neg, cats_neg = detect_emergency("Tôi ho nhẹ, không đau ngực và không khó thở.", upgraded=True)
    assert not is_em_neg
    assert cats_neg == []

    # Crisis triggers emergency + crisis addendum
    formatted = format_answer(REFUSAL, "Tôi cảm thấy tuyệt vọng và muốn tự tử", upgraded_guardrails=True)
    assert formatted.startswith(EMERGENCY)
    assert CRISIS_ADDENDUM in formatted
    assert formatted.endswith(WARNING)


def test_citation_verification_and_stripping():
    audit = verify_and_clean_citations(
        "Bạn có thể tiêm cách nhau 14 ngày [1, 2] và theo dõi thêm [9].", num_sources=5
    )
    assert audit.valid_indices == [1, 2]
    assert audit.invalid_indices == [9]
    assert audit.has_valid_citations is True
    assert "[9]" not in audit.cleaned_text
    assert "[1][2]" in audit.cleaned_text
    assert audit.citation_precision == round(2 / 3, 4)


def test_calibrate_min_score_separates_distributions():
    cal = calibrate_min_score(
        in_scope_scores=[0.62, 0.68, 0.71, 0.75, 0.80],
        out_of_scope_scores=[0.31, 0.35, 0.42, 0.48],
    )
    thr = cal["recommended_min_score"]
    assert 0.48 < thr < 0.62
    assert cal["metrics_at_threshold"]["tpr_in_scope"] == 1.0
    assert cal["metrics_at_threshold"]["tnr_oos_refusal"] == 1.0


def test_rag_repairs_missing_citations_and_fails_closed():
    hits = [
        {"doc_id": "train-1", "url": "https://example.com/1", "text": "Tiêm cách nhau 14 ngày.", "score": 0.78}
    ]
    retriever = _FakeRetriever(hits)

    # Case 1: First response misses citations -> repair adds [1] -> accepted.
    calls = {"n": 0}

    def chat_repair_success(model, messages, options):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Bạn có thể tiêm cách nhau 14 ngày."
        return "Bạn có thể tiêm cách nhau 14 ngày [1]."

    rag_ok = ViCareRAG(EMBEDDINGGEMMA2_CONFIG, retriever, chat_fn=chat_repair_success)
    res_ok = rag_ok.ask("Tiêm viêm gan B cách Covid-19 bao lâu?")
    assert not res_ok["refused"]
    assert calls["n"] == 2
    assert res_ok["citation_audit"]["repaired"] is True
    assert res_ok["sources"][0]["cited"] is True

    # Case 2: Repair still has no valid citations -> fails closed with REFUSAL.
    rag_fail = ViCareRAG(
        EMBEDDINGGEMMA2_CONFIG,
        retriever,
        chat_fn=lambda m, msgs, opts: "Câu trả lời không có nguồn trích dẫn.",
    )
    res_fail = rag_fail.ask("Tiêm viêm gan B cách Covid-19 bao lâu?")
    assert res_fail["refused"] is True
    assert res_fail["refusal_reason"] == "missing_valid_citations"
    assert REFUSAL in res_fail["answer"]
