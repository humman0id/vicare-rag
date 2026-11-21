"""Gradio chat demo UI for ViCare RAG with cited sources, status footer, and safety banners."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from vicare.api import _build_default_rag
from vicare.rag import ViCareRAG

DEMO_EXAMPLES = [
    ["Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?"],
    ["Trẻ bị sốt cao kèm co giật thì phải làm gì?"],
    ["Tôi bị ho nhẹ vài ngày, không đau ngực và không khó thở, có cần đi khám không?"],
    ["Thủ đô của nước Úc là gì?"],
]

_REFUSAL_REASON_LABELS = {
    "below_min_score": "điểm tương đồng dưới ngưỡng (ngoài phạm vi kho tài liệu)",
    "no_hits": "không có tài liệu nào trong index",
    "llm_refusal": "LLM xác định tài liệu truy xuất chưa đủ căn cứ",
    "missing_valid_citations": "thiếu trích dẫn nguồn hợp lệ sau bước kiểm tra",
    "llm_refusal_after_citation_repair": "LLM từ chối sau bước kiểm tra trích dẫn",
}


def format_sources_markdown(result: dict[str, Any]) -> str:
    """Format retrieved sources with citation status and similarity scores."""
    sources = result.get("sources") or []
    if not sources:
        return ""
    lines = ["\n\n**Tài liệu tham khảo:**"]
    for src in sources:
        idx = src["index"]
        url = str(src.get("url") or "")
        score = src.get("score")
        cited_badge = " ✓ *được trích dẫn*" if src.get("cited") else ""
        score_str = f" (score: `{score:.3f}`)" if isinstance(score, (int, float)) else ""
        if url.startswith(("https://", "http://")):
            lines.append(f"- [{idx}] {url}{score_str}{cited_badge}")
        else:
            lines.append(f"- [{idx}] `{src['doc_id']}`{score_str}{cited_badge}")
    return "\n".join(lines)


def format_status_footer(result: dict[str, Any]) -> str:
    """Render a compact transparency footer showing top score, threshold, and refusal reason."""
    parts: list[str] = []
    top_score = result.get("top_score")
    min_score = result.get("min_score")
    if isinstance(top_score, (int, float)) and isinstance(min_score, (int, float)):
        parts.append(f"Top score: `{top_score:.3f}` / ngưỡng `{min_score:.3f}`")
    if result.get("refused"):
        reason_key = result.get("refusal_reason") or "unknown"
        reason_vi = _REFUSAL_REASON_LABELS.get(reason_key, reason_key)
        parts.append(f"Trạng thái: từ chối ({reason_vi})")
    else:
        audit = result.get("citation_audit") or {}
        valid_ids = audit.get("valid_indices") or []
        if valid_ids:
            tags = ", ".join(f"[{i}]" for i in valid_ids)
            parts.append(f"Trích dẫn hợp lệ: {tags}")
    if not parts:
        return ""
    return "\n\n---\n*" + " · ".join(parts) + "*"


def build_demo(rag: ViCareRAG) -> gr.ChatInterface:
    """Build a Gradio ChatInterface bound to a ViCareRAG engine."""

    def chat_fn(message: str, history: list[Any]) -> str:
        result = rag.ask(message)
        return (
            result["answer"]
            + format_sources_markdown(result)
            + format_status_footer(result)
        )

    mode = "Hybrid BM25 + Vector RRF" if rag.config.use_hybrid_bm25 else "Vector-only"
    return gr.ChatInterface(
        fn=chat_fn,
        title="ViCare V2 – Trợ lý hỏi đáp sức khỏe tiếng Việt (DEMO HỌC THUẬT)",
        description=(
            f"**Embedding:** `{rag.config.embedding_model}` ({mode}) | "
            f"**LLM:** `{rag.config.llm_model}` | "
            f"**MIN_SCORE:** `{rag.config.min_score:.3f}` — "
            "Không nhập thông tin cá nhân hoặc dùng để chẩn đoán/điều trị y tế."
        ),
        examples=DEMO_EXAMPLES,
    )


def main() -> None:
    rag = _build_default_rag()
    demo = build_demo(rag)
    share = os.environ.get("VICARE_GRADIO_SHARE", "false").lower() == "true"
    demo.launch(share=share, show_error=True)


if __name__ == "__main__":
    main()
