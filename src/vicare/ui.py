"""Gradio chat demo UI for ViCare RAG with source citations and safety banners."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from vicare.api import _build_default_rag
from vicare.rag import ViCareRAG

DEMO_EXAMPLES = [
    ["Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?"],
    ["Trẻ bị sốt cao kèm co giật thì phải làm gì?"],
    ["Thủ đô của nước Úc là gì?"],
]


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


def build_demo(rag: ViCareRAG) -> gr.ChatInterface:
    """Build a Gradio ChatInterface bound to a ViCareRAG engine."""

    def chat_fn(message: str, history: list[Any]) -> str:
        result = rag.ask(message)
        return result["answer"] + format_sources_markdown(result)

    return gr.ChatInterface(
        fn=chat_fn,
        title="ViCare – Trợ lý hỏi đáp sức khỏe tiếng Việt (DEMO HỌC THUẬT)",
        description=(
            f"**Embedding:** `{rag.config.embedding_model}` | "
            f"**LLM:** `{rag.config.llm_model}` | "
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
