"""Unit tests for FastAPI endpoints using an injected fake RAG engine."""

from fastapi.testclient import TestClient

from vicare.api import create_app
from vicare.config import EMBEDDINGGEMMA2_CONFIG
from vicare.guardrails import EMERGENCY, REFUSAL
from vicare.rag import ViCareRAG


class _FakeRetriever:
    def retrieve(self, question: str, k: int = 5):
        if "Úc" in question:
            return [{"doc_id": "d-oos", "url": "", "text": "Nội dung y tế", "score": 0.30}]
        return [
            {
                "doc_id": "train-5104",
                "url": "https://vnexpress.net/example",
                "text": "Có thể tiêm cách nhau 14 ngày.",
                "score": 0.82,
            }
        ][:k]


def _make_client() -> TestClient:
    rag = ViCareRAG(
        EMBEDDINGGEMMA2_CONFIG,
        _FakeRetriever(),
        chat_fn=lambda m, msgs, opts: "Theo tài liệu, bạn có thể tiêm cách nhau 14 ngày [1].",
    )
    return TestClient(create_app(rag_engine=rag))


def test_health_and_meta():
    client = _make_client()
    assert client.get("/health").json() == {"status": "ok", "initialized": True}
    meta = client.get("/meta").json()
    assert meta["config"]["embedding_model"] == "google/embeddinggemma-2"


def test_ask_in_scope_and_out_of_scope_and_emergency():
    client = _make_client()

    # In-scope
    r_in = client.post("/ask", json={"question": "Tiêm viêm gan B có tiêm Covid-19 được không?"})
    assert r_in.status_code == 200
    body_in = r_in.json()
    assert body_in["refused"] is False
    assert body_in["sources"][0]["cited"] is True
    assert "[1]" in body_in["answer"]

    # Out-of-scope refusal
    r_oos = client.post("/ask", json={"question": "Thủ đô của nước Úc là gì?"})
    assert r_oos.status_code == 200
    body_oos = r_oos.json()
    assert body_oos["refused"] is True
    assert body_oos["refusal_reason"] == "below_min_score"
    assert REFUSAL in body_oos["answer"]

    # Emergency warning still included even on refusal
    r_em = client.post("/ask", json={"question": "Người nhà bị bất tỉnh ở Úc"})
    assert r_em.status_code == 200
    body_em = r_em.json()
    assert body_em["emergency_triggered"] is True
    assert body_em["answer"].startswith(EMERGENCY)
