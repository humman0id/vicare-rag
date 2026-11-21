"""ViHealthQA dataset loading, PII redaction, corpus deduplication, and dev sampling."""

from __future__ import annotations

import random
import re
from typing import Any

from datasets import load_dataset

# Matches common Vietnamese phone numbers and email addresses (identical to baseline).
PII_PATTERN = re.compile(
    r"(?<!\w)(?:\+?84|0)\d{9,10}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
)


def mask_pii(text: str) -> str:
    """Redact phone numbers and email addresses from raw text."""
    return PII_PATTERN.sub("[đã ẩn]", str(text or "")).strip()


def load_vihealthqa(
    dataset_name: str = "tarudesu/ViHealthQA",
    dev_seed: int = 2026,
    dev_size: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Load ViHealthQA and return (corpus, dev, gold_map).

    Matches the exact corpus construction and dev sampling logic of
    ViCare_Bootcamp24h.ipynb so retrieval/generation benchmarks remain
    strictly apples-to-apples comparable against the official baseline.
    """
    ds = load_dataset(dataset_name)
    needed_columns = {"id", "question", "answer", "link"}
    if "validation" not in ds or "train" not in ds:
        raise ValueError("Dataset must contain 'train' and 'validation' splits.")
    for split in ds:
        missing = needed_columns - set(ds[split].column_names)
        if missing:
            raise ValueError(f"Split '{split}' is missing columns: {missing}")

    text2id: dict[str, str] = {}
    corpus: list[dict[str, Any]] = []
    gold: dict[str, str] = {}

    for split in ds:
        for row in ds[split]:
            text = mask_pii(row.get("answer") or "")
            if not text:
                continue
            key = f"{split}-{row['id']}"
            if text not in text2id:
                text2id[text] = key
                corpus.append(
                    {
                        "doc_id": key,
                        "url": str(row.get("link") or ""),
                        "text": text,
                    }
                )
            gold[key] = text2id[text]

    if not corpus:
        raise ValueError("No valid answer documents found to build corpus.")

    rng = random.Random(dev_seed)
    validation_rows = list(ds["validation"])
    selected_rows = rng.sample(validation_rows, k=min(dev_size, len(validation_rows)))
    dev = [
        {
            "id": f"d{i:03d}",
            "question": row["question"],
            "answer": row["answer"],
            "gold_doc_ids": [gold[f"validation-{row['id']}"]],
        }
        for i, row in enumerate(selected_rows)
        if f"validation-{row['id']}" in gold and str(row.get("question") or "").strip()
    ]
    if not dev:
        raise ValueError("Failed to create dev split; verify validation data.")

    return corpus, dev, gold
