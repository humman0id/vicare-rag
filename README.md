# ViCare RAG — Vietnamese Health Q&A Assistant (AI4LIFE × Rikkei AI Challenge 24h)

> **Safety Notice:** ViCare is an **academic demo** built on [ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA). It is **not** a clinical diagnosis, emergency triage, or medical treatment tool. Do not enter personally identifiable health information or rely on it for medical decisions.

## Overview

This repository upgrades the official **ViCare Bootcamp 24h** baseline RAG pipeline into a modular Python package, FastAPI service, Gradio web UI, and free **Google Colab (T4 GPU)** submission notebook (`notebooks/ViCare_ai4life_submission.ipynb`).

### Planned Improvements Over Baseline

| Component | Official Baseline (`ViCare_Bootcamp24h`) | Upgraded ViCare (`vicare-rag`) |
|---|---|---|
| **Embedding Model** | `intfloat/multilingual-e5-small` (118M, 384d, 512 ctx) | `google/embeddinggemma-2` (270M text-only selective load, 768d / MRL, 8,192 ctx, `QuestionAnswering` / `Document` task prefixes, automatic `float32` on free T4 / `bfloat16` on L4/A100) + side-by-side baseline comparison |
| **Vector Store** | ChromaDB (L2, per-doc dedup) | ChromaDB with config-hashed signatures, latency & memory telemetry |
| **Prompt & Citations** | Basic system prompt (`[số]` requested, unverified) | Strict grounding prompt + deterministic `[k]` citation validator, hallucinated-citation stripper, and retry/repair guard |
| **Refusal Calibration** | Hardcoded `MIN_SCORE = 0.86` | Per-model threshold calibration on in-scope `dev` vs out-of-scope probe sets (critical when switching embedding score scales) |
| **Emergency Guardrails** | 8 substring keywords | Expanded Vietnamese red-flag symptoms + diacritic-normalized & negation-aware matching (e.g., avoids false alarm on *"không đau ngực"*) |
| **Packaging & Demo** | Single Colab notebook | Modular `src/vicare/` + FastAPI (`api.py`) + Gradio UI (`ui.py`) + free Colab T4 benchmark notebook |

## Repository Structure

```text
vicare-rag/
├── docs/
│   └── challenge_brief.md                  # Challenge rules, deadline, baseline numbers, submission checklist
├── notebooks/
│   ├── ViCare_Bootcamp24h_reference.ipynb  # Official baseline guide notebook (unmodified reference)
│   └── ViCare_ai4life_submission.ipynb     # Free Colab T4 submission notebook (synced with Google Drive)
├── src/vicare/
│   ├── config.py                           # Baseline & EmbeddingGemma 2 configuration profiles
│   ├── data.py                             # ViHealthQA loader, PII redaction, fixed 100-q dev split (seed=2026)
│   ├── chunking.py                         # Word-based chunking utilities
│   ├── embeddings.py                       # Unified E5 & EmbeddingGemma 2 embedder (text-only, T4 FP32 / L4 BF16)
│   ├── retriever.py                        # ChromaDB indexer & retriever with telemetry
│   ├── guardrails.py                       # Emergency detection, refusal calibration, strict citation verifier
│   ├── rag.py                              # End-to-end ViCareRAG pipeline with Ollama
│   ├── evaluation.py                       # Fair Recall@K, MRR@10, Judge@N, OOS refusal & trade-off comparison
│   ├── api.py                              # FastAPI REST service
│   └── ui.py                               # Gradio interactive demo
├── scripts/
│   ├── run_benchmark.py                    # Run side-by-side baseline vs EmbeddingGemma 2 benchmarks
│   └── sync_colab.py                       # Sync submission notebook to Google Drive
└── tests/                                  # Unit tests for chunking, guardrails, and API
```

## Quick Start

### 1. Local Development (CPU / API / UI / Tests)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Start the FastAPI backend:

```bash
uvicorn vicare.api:app --host 0.0.0.0 --port 8000
```

Start the Gradio UI:

```bash
python -m vicare.ui
```

### 2. Google Colab (Free T4 GPU — Judge-Reproducible)

Open `notebooks/ViCare_ai4life_submission.ipynb` in Google Colab with a free **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`) and run all cells top-to-bottom:
- Because T4 does not support native `bfloat16` and EmbeddingGemma 2 activations overflow `float16`, the pipeline automatically runs EmbeddingGemma 2 in **`float32`** on T4 (and automatically switches to **`bfloat16`** if run on L4/A100).
- The notebook unloads `multilingual-e5-small` from VRAM after the baseline benchmark before loading `google/embeddinggemma-2`, keeping peak GPU memory comfortably inside T4's 15 GB limit alongside Ollama `qwen2.5:3b`.

## Baseline Reference Metrics (from Official Guide Notebook)

Evaluated on the fixed 100-question `dev` split (`random.Random(2026)` from `ViHealthQA` validation, full 9,895-doc corpus / 11,227 chunks):

| Metric | Official Baseline (`multilingual-e5-small` + `qwen2.5:3b`) |
|---|---|
| `Recall@1` | 36.0% |
| `Recall@3` | 46.0% |
| `Recall@5` | 50.0% |
| `Recall@10` | 54.0% |
| `Coverage` | 100.0% |
| `Judge@10` | 70.0% |
| `Refusal` | 30.0% |
