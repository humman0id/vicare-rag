"""Fair retrieval, generation, citation, guardrail, and trade-off evaluation suite."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from vicare.guardrails import (
    DEFAULT_OUT_OF_SCOPE_PROBES,
    REFUSAL,
    calibrate_min_score,
)
from vicare.rag import ChatFn, ViCareRAG, default_ollama_chat

K_VALUES = (1, 3, 5, 10)
MRL_DIMS = (768, 512, 256, 128)

JUDGE_PROMPT = """Bạn là giám khảo đối chiếu câu trả lời với đáp án tham khảo.
CÂU HỎI: {q}
ĐÁP ÁN THAM KHẢO: {gold}
CÂU TRẢ LỜI CẦN CHẤM: {pred}
Chấm 1 nếu câu trả lời thể hiện đúng ý chính theo đáp án tham khảo,
ngược lại chấm 0. Nếu là từ chối khi đáp án có thể trả lời, chấm 0.
Chỉ trả lời một ký tự 0 hoặc 1, không giải thích."""


def evaluate_retrieval(
    retriever: Any,
    questions: list[dict[str, Any]],
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, Any]:
    """Compute Recall@K, MRR@maxK, Coverage, and query latency on fixed dev questions."""
    if not questions:
        raise ValueError("No questions provided for retrieval evaluation.")

    correct = {k: 0 for k in k_values}
    mrr_sum = 0.0
    covered = 0
    max_k = max(k_values)

    t0 = time.perf_counter()
    for q in questions:
        gold_ids = set(q["gold_doc_ids"])
        covered += int(bool(gold_ids & retriever.indexed_doc_ids))
        found_ids = [hit["doc_id"] for hit in retriever.retrieve(q["question"], k=max_k)]
        for k in k_values:
            correct[k] += int(bool(gold_ids & set(found_ids[:k])))
        for rank, doc_id in enumerate(found_ids, 1):
            if doc_id in gold_ids:
                mrr_sum += 1.0 / rank
                break
    elapsed_s = time.perf_counter() - t0

    n = len(questions)
    result: dict[str, Any] = {f"recall@{k}": correct[k] / n for k in k_values}
    result[f"mrr@{max_k}"] = mrr_sum / n
    result["coverage"] = covered / n
    result["n_questions"] = n
    result["avg_query_ms"] = round((elapsed_s / n) * 1000, 2)
    return result


def evaluate_threshold_calibration(
    retriever: Any,
    dev_questions: list[dict[str, Any]],
    oos_questions: tuple[str, ...] = DEFAULT_OUT_OF_SCOPE_PROBES,
    n_in_scope: int = 30,
) -> dict[str, Any]:
    """Measure in-scope vs out-of-scope max dense similarity distributions and calibrate MIN_SCORE."""
    top_k = getattr(getattr(retriever, "config", None), "top_k", 5)
    in_scores = [
        max(float(h["score"]) for h in hits)
        for q in dev_questions[:n_in_scope]
        if (hits := retriever.retrieve(q["question"], k=top_k))
    ]
    oos_scores = [
        max(float(h["score"]) for h in hits)
        for q in oos_questions
        if (hits := retriever.retrieve(q, k=top_k))
    ]
    return calibrate_min_score(in_scores, oos_scores)


def judge_answer(
    question: str,
    gold_answer: str,
    prediction: str,
    llm_model: str = "qwen2.5:3b",
    chat_fn: ChatFn | None = None,
) -> int | None:
    """Run LLM-as-a-judge (matching ViCare_Bootcamp24h.ipynb rubric)."""
    fn = chat_fn or default_ollama_chat
    raw = fn(
        llm_model,
        [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    q=question,
                    gold=str(gold_answer or "")[:1500],
                    pred=prediction,
                ),
            }
        ],
        {"temperature": 0, "num_predict": 8},
    )
    return int(raw) if re.fullmatch(r"[01]", raw.strip()) else None


def evaluate_generation(
    rag: ViCareRAG,
    dev_questions: list[dict[str, Any]],
    n_judge: int = 10,
    judge_llm_model: str | None = None,
    judge_chat_fn: ChatFn | None = None,
    oos_questions: tuple[str, ...] = DEFAULT_OUT_OF_SCOPE_PROBES[:5],
) -> dict[str, Any]:
    """Evaluate Judge@N, refusal rate, citation compliance, latency, and OOS refusal rate."""
    if not isinstance(n_judge, int) or n_judge < 1:
        raise ValueError("n_judge must be an integer >= 1.")

    subset = dev_questions[:n_judge]
    judge_model = judge_llm_model or rag.config.llm_model

    scores: list[int | None] = []
    refusals = 0
    non_refusal_count = 0
    cited_non_refusal = 0
    precision_values: list[float] = []
    latencies: list[float] = []
    repaired_count = 0

    for q in subset:
        res = rag.ask(q["question"])
        pred = res["answer"]
        latencies.append(float(res.get("latency_ms", 0.0)))
        is_ref = bool(res.get("refused")) or (REFUSAL in pred)
        refusals += int(is_ref)
        if not is_ref:
            non_refusal_count += 1
            audit = res.get("citation_audit") or {}
            if audit.get("has_valid_citations"):
                cited_non_refusal += 1
            precision_values.append(float(audit.get("citation_precision", 0.0)))
            if audit.get("repaired"):
                repaired_count += 1
        scores.append(
            judge_answer(
                q["question"],
                q["answer"],
                pred,
                llm_model=judge_model,
                chat_fn=judge_chat_fn,
            )
        )

    # Check out-of-scope refusal accuracy on probe questions.
    oos_refused = 0
    for oos_q in oos_questions:
        oos_res = rag.ask(oos_q)
        if oos_res.get("refused") or (REFUSAL in oos_res["answer"]):
            oos_refused += 1

    return {
        "judge_rate": sum(s == 1 for s in scores) / len(subset),
        "refusal_rate": refusals / len(subset),
        "oos_refusal_rate": (oos_refused / len(oos_questions)) if oos_questions else None,
        "citation_compliance_rate": (
            cited_non_refusal / non_refusal_count if non_refusal_count > 0 else 1.0
        ),
        "avg_citation_precision": (
            round(sum(precision_values) / len(precision_values), 4)
            if precision_values
            else 1.0
        ),
        "citation_repairs": repaired_count,
        "n_questions": len(subset),
        "invalid_judgments": sum(s is None for s in scores),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
    }


def format_comparison_table(
    baseline: dict[str, Any],
    improved: dict[str, Any],
) -> str:
    """Format the official Before / After comparison table (in percentage points)."""
    lines = [
        "BẢNG SO SÁNH TRƯỚC / SAU",
        f"{'Chỉ số':<24} {'Trước':>10} {'Sau':>10} {'Thay đổi':>14}",
        "-" * 62,
    ]

    def _row(label: str, before: float, after: float) -> str:
        diff = (after - before) * 100.0
        return f"{label:<24} {before:>9.1%} {after:>9.1%} {diff:>+11.1f} đ.%"

    b_ret = baseline.get("retrieval", {})
    i_ret = improved.get("retrieval", {})
    for k in K_VALUES:
        key = f"recall@{k}"
        if key in b_ret and key in i_ret:
            lines.append(_row(key, b_ret[key], i_ret[key]))
    if "mrr@10" in b_ret and "mrr@10" in i_ret:
        lines.append(_row("mrr@10", b_ret["mrr@10"], i_ret["mrr@10"]))
    if "coverage" in b_ret and "coverage" in i_ret:
        lines.append(_row("coverage", b_ret["coverage"], i_ret["coverage"]))

    b_judge = baseline.get("judge")
    i_judge = improved.get("judge")
    if b_judge and i_judge and b_judge.get("n_questions") == i_judge.get("n_questions"):
        lines.append(_row(f"judge@{b_judge['n_questions']}", b_judge["judge_rate"], i_judge["judge_rate"]))
        lines.append(_row("in_scope_refusal", b_judge["refusal_rate"], i_judge["refusal_rate"]))
        if "oos_refusal_rate" in b_judge and "oos_refusal_rate" in i_judge:
            if b_judge["oos_refusal_rate"] is not None and i_judge["oos_refusal_rate"] is not None:
                lines.append(_row("oos_refusal", b_judge["oos_refusal_rate"], i_judge["oos_refusal_rate"]))
        if "citation_compliance_rate" in b_judge and "citation_compliance_rate" in i_judge:
            lines.append(
                _row(
                    "citation_compliance",
                    b_judge["citation_compliance_rate"],
                    i_judge["citation_compliance_rate"],
                )
            )
    return "\n".join(lines)


def evaluate_matryoshka_ablation(
    doc_embeddings: np.ndarray,
    doc_ids: list[str],
    query_embeddings: np.ndarray,
    questions: list[dict[str, Any]],
    dims: tuple[int, ...] = MRL_DIMS,
    k_values: tuple[int, ...] = K_VALUES,
) -> list[dict[str, Any]]:
    """Evaluate Matryoshka (MRL) vector truncation without re-encoding.

    Slices full-dimensional embeddings to each target dimension, re-normalizes to unit
    L2 length (required by EmbeddingGemma 2), deduplicates per doc_id, and computes
    exact Recall@K, MRR@maxK, and raw float32 vector storage in MB.
    """
    full_dim = int(doc_embeddings.shape[1])
    n_chunks = int(doc_embeddings.shape[0])
    n_queries = len(questions)
    max_k = max(k_values)
    candidate_pool = min(n_chunks, max(max_k * 8, 60))

    results: list[dict[str, Any]] = []
    for dim in dims:
        if dim > full_dim:
            continue
        t0 = time.perf_counter()
        d_slice = doc_embeddings[:, :dim].astype(np.float32, copy=False)
        q_slice = query_embeddings[:, :dim].astype(np.float32, copy=False)
        d_norm = d_slice / np.clip(np.linalg.norm(d_slice, axis=1, keepdims=True), 1e-12, None)
        q_norm = q_slice / np.clip(np.linalg.norm(q_slice, axis=1, keepdims=True), 1e-12, None)

        sim_matrix = q_norm @ d_norm.T

        correct = {k: 0 for k in k_values}
        mrr_sum = 0.0
        for qi, q in enumerate(questions):
            row = sim_matrix[qi]
            top_idx = np.argpartition(row, -candidate_pool)[-candidate_pool:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]

            found_ids: list[str] = []
            seen: set[str] = set()
            for idx in top_idx:
                did = doc_ids[int(idx)]
                if did in seen:
                    continue
                seen.add(did)
                found_ids.append(did)
                if len(found_ids) >= max_k:
                    break

            gold_ids = set(q["gold_doc_ids"])
            for k in k_values:
                correct[k] += int(bool(gold_ids & set(found_ids[:k])))
            for rank, did in enumerate(found_ids, 1):
                if did in gold_ids:
                    mrr_sum += 1.0 / rank
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        entry: dict[str, Any] = {
            "dim": dim,
            "compression": f"{full_dim / dim:.1f}x",
            "vector_ram_mb": round((n_chunks * dim * 4) / (1024 * 1024), 2),
            "search_ms_total": round(elapsed_ms, 1),
        }
        for k in k_values:
            entry[f"recall@{k}"] = correct[k] / n_queries
        entry[f"mrr@{max_k}"] = mrr_sum / n_queries
        results.append(entry)
    return results


def format_matryoshka_table(rows: list[dict[str, Any]]) -> str:
    """Format MRL truncation ablation results into a compact comparison table."""
    lines = [
        "BẢNG THỰC NGHIỆM MATRYOSHKA (MRL TRUNCATION — DENSE-ONLY, KHÔNG CẦN ENCODE LẠI)",
        f"{'Dim':<6} {'Nén':>6} {'RAM Vector':>12} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR@10':>8}",
        "-" * 72,
    ]
    for r in rows:
        lines.append(
            f"{r['dim']:<6} {r['compression']:>6} {r['vector_ram_mb']:>9.2f} MB "
            f"{r['recall@1']:>7.1%} {r['recall@3']:>7.1%} "
            f"{r['recall@5']:>7.1%} {r['recall@10']:>7.1%} {r['mrr@10']:>7.1%}"
        )
    return "\n".join(lines)


def save_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist structured JSON benchmark artifacts with UTF-8 Vietnamese support."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
