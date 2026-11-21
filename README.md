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

## Verified Benchmark Results (Free Google Colab Tesla T4 — V2.2)

Evaluated apples-to-apples on the fixed 100-question `dev` split (`random.Random(2026)` from `ViHealthQA` validation, full 9,895-doc corpus / 11,227 chunks) with `qwen2.5:3b` (`temperature=0`, `seed=42`):

| Metric | Baseline (`multilingual-e5-small`) | Upgraded V2.2 (`google/embeddinggemma-2` + Hybrid BM25 RRF) | Delta |
|---|---:|---:|---:|
| `Recall@1` | `37.0%` | `34.0%` *(Dense `ef=200`: `35.0%`; MRL `512d`: `37.0%`)* | `-3.0 pp` |
| `Recall@3` | `47.0%` | **`50.0%`** | **`+3.0 pp`** |
| `Recall@5` | `51.0%` | **`54.0%`** *(Dense `ef=200`: **`55.0%`**)* | **`+3.0 pp`** |
| `Recall@10` | `55.0%` | **`58.0%`** *(Dense `ef=200`: **`60.0%`**)* | **`+3.0 pp`** |
| `MRR@10` | `42.7%` | **`42.8%`** *(Dense `ef=200`: **`43.2%`**)* | `+0.1 pp` |
| `Coverage` | `100.0%` | `100.0%` | `+0.0 pp` |
| `Judge@10` | `80.0%` | **`80.0%`** | `+0.0 pp` |
| `in_scope_refusal` | `20.0%` | **`10.0%`** | **`-10.0 pp`** |
| `oos_refusal` | `100.0%` | **`100.0%`** | `+0.0 pp` |
| `citation_compliance` | `12.5%` | **`100.0%`** *(`88.9%` first-pass + `1` auto-repair)* | **`+87.5 pp`** |

### Honest Resource Trade-Offs (Tesla T4, `float32`)

| Resource | Baseline (`multilingual-e5-small`) | Upgraded V2.2 (`google/embeddinggemma-2` text-only) |
|---|---:|---:|
| Active Parameters | `117.7M` | `271.0M` *(vision/audio encoders skipped)* |
| Vector Dimension | `384` | `768` *(MRL: `512 / 256 / 128`)* |
| Vector Index Time (11,227 chunks) | `71.3 s` | `266.0 s` |
| Avg Query Latency | `22.4 ms` | `205.4 ms` *(Dense `ef=200`: `78.5 ms`)* |
| Isolated Model VRAM | `449.6 MB` | `1034.7 MB` |
| Calibrated `MIN_SCORE` | `0.86` (fixed) | `0.6707` (`100%` TPR / `100%` TNR, margin `+0.0717`) |

### Matryoshka (MRL) Truncation Ablation (Zero Re-Encoding)

| Dim | Compression | Vector RAM | `R@1` | `R@3` | `R@5` | `R@10` | `MRR@10` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `768` | `1.0x` | `32.89 MB` | `35.0%` | `48.0%` | `55.0%` | `60.0%` | `43.2%` |
| `512` | `1.5x` | `21.93 MB` | **`37.0%`** | `45.0%` | `52.0%` | `58.0%` | **`43.3%`** |
| `256` | `3.0x` | `10.96 MB` | `32.0%` | `45.0%` | `52.0%` | `57.0%` | `40.2%` |
| `128` | `6.0x` | `5.48 MB` | `26.0%` | `40.0%` | `46.0%` | `51.0%` | `34.1%` |

### Key Caveats & Limitations
- `citation_compliance = 100%` reflects **structural validity** (in-range `[k]` tags guaranteed by fail-closed enforcement after `88.9%` first-pass natural citation + `1` repair), not full NLI semantic attribution verification.
- Emergency detection extends the official baseline's 8 hardcoded keywords into a 5-category diacritic-insensitive, clause-bounded negation lexicon, but remains **rule/lexicon-based** rather than a clinical triage classifier.
- `ViHealthQA` evaluates closed-corpus retrieval and `Judge@10` uses a small 10-question sample; ViCare is strictly an academic demo, not a medical diagnosis or treatment tool.
