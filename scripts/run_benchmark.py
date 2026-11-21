#!/usr/bin/env python3
"""Run apples-to-apples Baseline (multilingual-e5-small) vs EmbeddingGemma 2 benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from vicare.chunking import build_chunks
from vicare.config import BASELINE_CONFIG, EMBEDDINGGEMMA2_CONFIG, ViCareConfig
from vicare.data import load_vihealthqa
from vicare.embeddings import ViCareEmbedder
from vicare.evaluation import (
    evaluate_generation,
    evaluate_retrieval,
    evaluate_threshold_calibration,
    format_comparison_table,
    save_json_report,
)
from vicare.rag import ViCareRAG
from vicare.retriever import ViCareRetriever


def run_profile(
    config: ViCareConfig,
    corpus: list[dict],
    dev: list[dict],
    run_generation: bool,
    auto_calibrate: bool,
) -> dict:
    print(f"\n=== Running profile: {config.profile_name} ({config.embedding_model}) ===")
    chunks = build_chunks(corpus, chunk_size=config.chunk_size, overlap=config.overlap)
    embedder = ViCareEmbedder(config)
    retriever = ViCareRetriever(config, embedder, chunks)

    retrieval = evaluate_retrieval(retriever, dev)
    calibration = evaluate_threshold_calibration(retriever, dev)
    effective_config = config
    if auto_calibrate:
        effective_config = replace(config, min_score=calibration["recommended_min_score"])

    report = {
        "profile": effective_config.profile_name,
        "settings": {
            "embedding_model": effective_config.embedding_model,
            "chunk_size": effective_config.chunk_size,
            "overlap": effective_config.overlap,
            "n_chunks": effective_config.n_chunks,
            "min_score": effective_config.min_score,
            "llm_model": effective_config.llm_model,
            "truncate_dim": effective_config.truncate_dim,
            "query_prompt_name": effective_config.query_prompt_name,
        },
        "embedder_telemetry": embedder.telemetry(),
        "indexer_telemetry": retriever.telemetry(),
        "retrieval": retrieval,
        "calibration": calibration,
    }

    if run_generation:
        rag = ViCareRAG(effective_config, retriever)
        report["judge"] = evaluate_generation(
            rag, dev, n_judge=effective_config.n_judge
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ViCare Baseline vs EmbeddingGemma 2")
    parser.add_argument("--n-chunks", type=int, default=None, help="Optional subset size for quick dev runs")
    parser.add_argument("--n-judge", type=int, default=10, help="Number of dev questions for LLM-as-judge")
    parser.add_argument("--llm-model", type=str, default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--skip-generation", action="store_true", help="Run retrieval + calibration only")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Output directory for JSON reports")
    args = parser.parse_args()

    corpus, dev, _ = load_vihealthqa()
    base_cfg = replace(BASELINE_CONFIG, n_chunks=args.n_chunks, n_judge=args.n_judge, llm_model=args.llm_model)
    imp_cfg = replace(EMBEDDINGGEMMA2_CONFIG, n_chunks=args.n_chunks, n_judge=args.n_judge, llm_model=args.llm_model)

    baseline_report = run_profile(
        base_cfg, corpus, dev, run_generation=not args.skip_generation, auto_calibrate=False
    )
    improved_report = run_profile(
        imp_cfg, corpus, dev, run_generation=not args.skip_generation, auto_calibrate=True
    )

    save_json_report(args.out_dir / "vicare_baseline.json", baseline_report)
    save_json_report(
        args.out_dir / "vicare_comparison.json",
        {"baseline": baseline_report, "improved": improved_report},
    )
    print("\n" + format_comparison_table(baseline_report, improved_report))


if __name__ == "__main__":
    main()
