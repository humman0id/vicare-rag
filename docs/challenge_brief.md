# AI Challenge 24h — ViCare Project Brief

- **Organizers:** Viện AI4LIFE — ĐH Bách Khoa Hà Nội & Rikkei
- **Challenge:** AI Challenge 24h (Post-Workshop *AI Engineer Roadmap*)
- **Deadline:** **12:00 (noon), Monday 23/11/2026**
- **Official Materials Folder:** `https://drive.google.com/drive/folders/1Eai3oBqdK--ZDx2ZiUIGt33F8osOU_Cv`
- **Official Guide Notebook:** `ViCare_Bootcamp24h.ipynb` (`1gNdeJCpNC_SfxtK8xWoBqhNGC7V42n4d`)
- **Submission Colab Notebook:** `ViCare_ai4life_submission.ipynb` (`1oobFwEBHSVCTt2S8qAH-UNI63OraRCHN`)

## Problem & Baseline

Build and improve **ViCare**, a Vietnamese health information Q&A assistant using Retrieval-Augmented Generation (RAG) over [ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA) (`question`, `answer`, `link`).

### Official Baseline Stack & Numbers

- **Corpus:** 9,895 unique deduplicated answers (with regex PII redaction), chunked at `CHUNK_SIZE=200`, `OVERLAP=40` -> 11,227 chunks
- **Dev Set:** 100 fixed questions sampled via `random.Random(2026).sample(list(ds["validation"]), 100)`
- **Embedding:** `intfloat/multilingual-e5-small` (384d, L2-normalized, `passage: ` / `query: `)
- **Vector DB:** ChromaDB (`hnsw:space=l2`, `score = 1 - dist / 2`, deduplicated by `doc_id`)
- **LLM:** Ollama `qwen2.5:3b` (`temperature=0`, `num_predict=256`)
- **Guardrails:** `MIN_SCORE=0.86`, 8 emergency keywords, standard refusal + disclaimer
- **Official Baseline Results:**
  - `Recall@1`: `36.0%`
  - `Recall@3`: `46.0%`
  - `Recall@5`: `50.0%`
  - `Recall@10`: `54.0%`
  - `Coverage`: `100.0%`
  - `Judge@10`: `70.0%`
  - `Refusal`: `30.0%`

## Our Improvement Plan & Honest Trade-Off Analysis

1. **Embedding Upgrade (`google/embeddinggemma-2`) + Baseline Side-by-Side Comparison**
   - Load text-only backbone via selective encoder loading (`config_kwargs={"vision_config": None, "audio_config": None}` -> 270M active params instead of full 740M multimodal).
   - Enforce `bfloat16` on bf16-capable GPUs (Colab Pro L4/A100) and `float32` on CPU/T4 (never `float16`, which overflows activations in EmbeddingGemma 2).
   - Use asymmetric task prefixes: `prompt_name="QuestionAnswering"` (and optionally ablate `"SearchQuery"`) for queries, `prompt_name="Document"` for passages.
   - Benchmark against the original `intfloat/multilingual-e5-small` on the exact same 100-question `dev` split and report honest trade-offs: Recall@1/3/5/10, MRR@10, indexing time, query latency, memory/VRAM footprint, vector dimension (768d vs 384d, plus optional MRL 256d/512d check), and hardware/dependency caveats (`bfloat16`/`float32` precision requirement, `sentence-transformers>=6.1` + `transformers>=5.18`, and `torchvision` required by `AutoProcessor` even in text-only mode).
2. **Improved Prompting & Strict `[k]` Citation Enforcement**
   - Refined Vietnamese system prompt separating retrieved evidence from instructions and forbidding hallucinated facts/dosages/diagnoses.
   - Deterministic post-generation citation verifier: validates every `[k]` tag against `1..len(hits)`, strips out-of-range tags, verifies that non-refusal answers actually cite at least one source, and performs one targeted citation-repair pass if missing.
3. **Calibrated Refusal Threshold & Out-of-Scope Evaluation**
   - Because cosine score distributions differ sharply between `multilingual-e5-small` and `embeddinggemma-2`, calibrate `MIN_SCORE` per model using in-scope `dev` scores vs an expanded out-of-scope probe set (maximizing F1 / Youden's J) rather than blindly reusing `0.86`.
4. **Upgraded Vietnamese Emergency Guardrails**
   - Expanded red-flag symptom lexicon (cardiac, neurological/stroke, severe bleeding, respiratory distress, anaphylaxis, acute self-harm/crisis, severe pediatric fever) with diacritic-insensitive matching and negation checks (avoiding false positives on phrases like *"không đau ngực"*).
5. **Production Packaging & Demo**
   - Modular `src/vicare/` package, FastAPI server (`vicare.api`), Gradio chat UI (`vicare.ui`), automated unit tests, and self-contained Colab Pro submission notebook synced with Google Drive.

## Submission Checklist (2 Mandatory Steps)

1. **Facebook Group Post** in *Mê IT Nghiện Code* (`https://www.facebook.com/groups/meitnghiencode`):
   - Title: `[AI Challenge 24h] - [Họ và tên thí sinh / Tên nhóm]`
   - Media: 1–3 min demo screencast video OR >= 3 screenshots (UI, actual results, source code)
   - Required evidence: >= 1 valid in-scope example, >= 1 out-of-scope refusal example, Recall@5 before/after table, >= 1 identified limitation
   - Public GitHub & Colab links + hashtags `#AIChallenge24h #AI4LIFE #Rikkei`
2. **Certification Registration Form:**
   - `https://m4beo6fqhrl.sg.larksuite.com/share/base/form/shrlgB534MeoVcQaAjNI8oP2cBe`
