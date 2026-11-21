"""Safety guardrails, Vietnamese emergency detection, refusal calibration, and strict citation verification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Kept identical to ViCare_Bootcamp24h.ipynb for seamless evaluation compatibility.
REFUSAL = "Tôi không tìm thấy thông tin này trong kho tài liệu."
WARNING = "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn của nhân viên y tế."
EMERGENCY = (
    "⚠️ Nếu có dấu hiệu nguy hiểm hoặc cấp cứu, hãy gọi 115 (tại Việt Nam) "
    "hoặc số cấp cứu địa phương ngay."
)
CRISIS_ADDENDUM = (
    "⚠️ Nếu bạn hoặc người thân đang có ý định tự làm hại bản thân, hãy gọi ngay "
    "115 hoặc đường dây nóng hỗ trợ khẩn cấp và liên hệ người thân/cơ sở y tế gần nhất."
)

# Original 8 baseline emergency phrases.
BASELINE_EMERGENCY_WORDS: tuple[str, ...] = (
    "đau ngực",
    "khó thở",
    "co giật",
    "bất tỉnh",
    "ngất",
    "chảy máu nhiều",
    "đột quỵ",
    "ngộ độc",
)

# Expanded Vietnamese red-flag symptom lexicon grouped by clinical category.
UPGRADED_EMERGENCY_PATTERNS: dict[str, tuple[str, ...]] = {
    "cardiac_respiratory": (
        "đau ngực",
        "đau thắt ngực",
        "tức ngực dữ dội",
        "khó thở",
        "ngừng thở",
        "thở dốc",
        "tím tái",
        "môi tím",
    ),
    "neurological_stroke": (
        "co giật",
        "bất tỉnh",
        "mất ý thức",
        "hôn mê",
        "ngất",
        "đột quỵ",
        "liệt nửa người",
        "yếu liệt đột ngột",
        "méo miệng",
        "nói ngọng đột ngột",
    ),
    "hemorrhage_trauma": (
        "chảy máu nhiều",
        "xuất huyết ồ ạt",
        "nôn ra máu",
        "ói ra máu",
        "ho ra máu",
        "chấn thương sọ não",
    ),
    "allergy_toxicology": (
        "ngộ độc",
        "sốc phản vệ",
        "phù mặt",
        "sưng môi lưỡi",
        "uống nhầm hóa chất",
        "quá liều thuốc",
    ),
    "self_harm_crisis": (
        "tự tử",
        "tự sát",
        "muốn chết",
        "tự làm hại bản thân",
        "cắt tay tự tử",
    ),
}

# Negation markers immediately preceding a symptom (e.g., "không đau ngực", "chưa bị ngất").
_NEGATION_RE = re.compile(
    r"(?:^|[\s,.;:!?])(?:khong|chua|chua tung|khong co|khong bi|chua bi)\s+$"
)

# Matches bracketed citations such as [1], [2], [1, 2], [1,3].
_CITATION_GROUP_RE = re.compile(r"\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]")

DEFAULT_OUT_OF_SCOPE_PROBES: tuple[str, ...] = (
    "Thủ đô của nước Úc là gì?",
    "Cách nấu phở bò ngon chuẩn vị Hà Nội?",
    "Giá vàng SJC hôm nay bao nhiêu?",
    "Ai là tác giả của Truyện Kiều?",
    "Làm sao để học lập trình Python từ đầu?",
    "Đội tuyển nào vô địch World Cup 2022?",
    "Cách tính đạo hàm của hàm số lượng giác?",
    "So sánh iPhone 16 và Samsung Galaxy S25?",
    "Thời tiết Đà Nẵng tháng 11 như thế nào?",
    "Viết một bài thơ 4 câu về mùa thu Hà Nội.",
    "Lãi suất gửi tiết kiệm ngân hàng hiện nay là bao nhiêu?",
    "Cách sửa lỗi ModuleNotFoundError trong Python?",
    "Tóm tắt nội dung tiểu thuyết Số Đỏ của Vũ Trọng Phụng.",
    "Công thức làm bánh mì hoa cúc bằng nồi chiên không dầu?",
    "Quãng đường từ Hà Nội đến TP.HCM dài bao nhiêu km?",
)


def strip_vietnamese_diacritics(text: str) -> str:
    """Fold Vietnamese text to lowercase ASCII for resilient matching."""
    text = str(text or "").replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped.casefold()).strip()


def detect_emergency(
    question: str,
    upgraded: bool = True,
) -> tuple[bool, list[str]]:
    """Detect emergency red-flag symptoms in a Vietnamese health question.

    In baseline mode (`upgraded=False`), reproduces the exact substring check from
    ViCare_Bootcamp24h.ipynb. In upgraded mode (`upgraded=True`), performs
    diacritic-insensitive, word-boundary, negation-aware category matching.
    """
    raw = str(question or "")
    if not raw.strip():
        return False, []

    if not upgraded:
        folded = raw.casefold()
        matched = [w for w in BASELINE_EMERGENCY_WORDS if w in folded]
        return bool(matched), (["baseline_emergency"] if matched else [])

    norm_q = strip_vietnamese_diacritics(raw)
    matched_categories: list[str] = []

    for category, phrases in UPGRADED_EMERGENCY_PATTERNS.items():
        for phrase in phrases:
            norm_p = strip_vietnamese_diacritics(phrase)
            pattern = re.compile(rf"(?<!\w){re.escape(norm_p)}(?!\w)")
            for match in pattern.finditer(norm_q):
                prefix = norm_q[max(0, match.start() - 25) : match.start()]
                if _NEGATION_RE.search(prefix):
                    continue
                if category not in matched_categories:
                    matched_categories.append(category)
                break
            if category in matched_categories:
                break

    return bool(matched_categories), matched_categories


def format_answer(
    answer: str,
    question: str,
    upgraded_guardrails: bool = True,
) -> str:
    """Attach emergency banner (if triggered) and standard safety disclaimer."""
    emergency, categories = detect_emergency(question, upgraded=upgraded_guardrails)
    prefixes: list[str] = []
    if emergency:
        prefixes.append(EMERGENCY)
        if upgraded_guardrails and "self_harm_crisis" in categories:
            prefixes.append(CRISIS_ADDENDUM)

    body = answer.strip()
    if prefixes:
        return "\n\n".join(prefixes) + "\n\n" + body + "\n\n" + WARNING
    return body + "\n\n" + WARNING


@dataclass
class CitationAudit:
    """Audit result for inline [k] citations in an LLM answer."""

    valid_indices: list[int]
    invalid_indices: list[int]
    has_valid_citations: bool
    citation_precision: float
    cleaned_text: str


def verify_and_clean_citations(text: str, num_sources: int) -> CitationAudit:
    """Parse, validate, and clean inline [k] citations against 1..num_sources.

    - Supports single tags (`[1]`) and comma-grouped tags (`[1, 2]`).
    - Strips any out-of-range citation numbers (e.g., `[7]` when only 5 sources exist).
    - Computes citation precision = valid_mentions / total_mentions.
    """
    valid: list[int] = []
    invalid: list[int] = []

    def _replace_group(match: re.Match[str]) -> str:
        raw_nums = [int(n.strip()) for n in match.group(1).split(",") if n.strip()]
        kept: list[int] = []
        for n in raw_nums:
            if 1 <= n <= num_sources:
                valid.append(n)
                if n not in kept:
                    kept.append(n)
            else:
                invalid.append(n)
        if not kept:
            return ""
        return "".join(f"[{n}]" for n in kept)

    cleaned = _CITATION_GROUP_RE.sub(_replace_group, str(text or ""))
    # Tidy double spaces left after stripping an invalid citation.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip()

    total = len(valid) + len(invalid)
    precision = (len(valid) / total) if total > 0 else 0.0
    unique_valid = sorted(set(valid))
    unique_invalid = sorted(set(invalid))

    return CitationAudit(
        valid_indices=unique_valid,
        invalid_indices=unique_invalid,
        has_valid_citations=bool(unique_valid),
        citation_precision=round(precision, 4),
        cleaned_text=cleaned,
    )


def calibrate_min_score(
    in_scope_scores: list[float],
    out_of_scope_scores: list[float],
) -> dict[str, Any]:
    """Calibrate a refusal threshold separating in-scope from out-of-scope scores.

    Scans midpoints between sorted unique scores and picks the threshold
    maximizing F1 on in-scope acceptance (tie-breaking by Youden's J = TPR - FPR).
    """
    if not in_scope_scores or not out_of_scope_scores:
        raise ValueError("Both in_scope_scores and out_of_scope_scores must be non-empty.")

    all_scores = sorted(set(in_scope_scores + out_of_scope_scores))
    candidates = [all_scores[0] - 0.01]
    for a, b in zip(all_scores, all_scores[1:]):
        candidates.append((a + b) / 2.0)
    candidates.append(all_scores[-1] + 0.01)

    best_thr = candidates[0]
    best_key = (-1.0, -1.0, -1.0)
    best_stats: dict[str, float] = {}

    n_in = len(in_scope_scores)
    n_oos = len(out_of_scope_scores)

    for thr in candidates:
        tp = sum(s >= thr for s in in_scope_scores)
        fn = n_in - tp
        fp = sum(s >= thr for s in out_of_scope_scores)
        tn = n_oos - fp

        tpr = tp / n_in
        tnr = tn / n_oos
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
        youden_j = tpr + tnr - 1.0
        # Prefer thresholds that cleanly reject out-of-scope queries when tied.
        key = (round(youden_j, 6), round(f1, 6), round(tnr, 6))
        if key > best_key:
            best_key = key
            best_thr = thr
            best_stats = {
                "tpr_in_scope": round(tpr, 4),
                "tnr_oos_refusal": round(tnr, 4),
                "f1": round(f1, 4),
                "youden_j": round(youden_j, 4),
            }

    in_sorted = sorted(in_scope_scores)
    oos_sorted = sorted(out_of_scope_scores)
    p10_idx = max(0, int(0.10 * (len(in_sorted) - 1)))

    return {
        "recommended_min_score": round(best_thr, 4),
        "in_scope": {
            "count": n_in,
            "min": round(in_sorted[0], 4),
            "p10": round(in_sorted[p10_idx], 4),
            "mean": round(sum(in_sorted) / n_in, 4),
            "max": round(in_sorted[-1], 4),
        },
        "out_of_scope": {
            "count": n_oos,
            "min": round(oos_sorted[0], 4),
            "mean": round(sum(oos_sorted) / n_oos, 4),
            "max": round(oos_sorted[-1], 4),
        },
        "separation_margin": round(in_sorted[0] - oos_sorted[-1], 4),
        "metrics_at_threshold": best_stats,
    }
