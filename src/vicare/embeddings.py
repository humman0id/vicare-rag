"""Unified embedding wrapper supporting multilingual-e5-small and EmbeddingGemma 2."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from vicare.config import ViCareConfig


def select_safe_dtype(embedding_backend: str) -> torch.dtype:
    """Select a numerically safe dtype for the target embedding backend.

    EmbeddingGemma 2 activations exceed float16 dynamic range and silently produce
    NaNs or degraded vectors in float16. Always use bfloat16 on bf16-capable GPUs
    (e.g., L4/A100) and float32 elsewhere (free Colab T4, CPU).
    """
    if embedding_backend == "embeddinggemma2":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return torch.float32


class ViCareEmbedder:
    """Handles model loading, task instruction prefixes, MRL truncation, and telemetry."""

    def __init__(self, config: ViCareConfig, device: str | None = None) -> None:
        self.config = config
        self.dtype = select_safe_dtype(config.embedding_backend)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        t0 = time.perf_counter()
        if config.embedding_backend == "embeddinggemma2":
            model_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
            config_kwargs: dict[str, Any] = {}
            if config.text_only_encoders:
                # Selective encoder loading: keep only the 270M text backbone
                # and skip vision (170M) + audio (300M) encoders.
                config_kwargs = {"vision_config": None, "audio_config": None}
            self.model = SentenceTransformer(
                config.embedding_model,
                device=self.device,
                model_kwargs=model_kwargs,
                config_kwargs=config_kwargs,
            )
        else:
            self.model = SentenceTransformer(config.embedding_model, device=self.device)

        self.load_time_s = round(time.perf_counter() - t0, 3)
        self.model_vram_mb = (
            round(max(0, torch.cuda.memory_allocated() - mem_before) / (1024 * 1024), 1)
            if torch.cuda.is_available()
            else None
        )
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.embedding_dim = len(self.embed_query("kiểm tra"))

    def unload(self) -> None:
        """Release GPU VRAM cleanly between sequential model benchmarks (critical on T4)."""
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed_passages(
        self,
        texts: list[str],
        show_progress_bar: bool | None = None,
    ) -> list[list[float]]:
        """Encode corpus passages into L2-normalized vectors."""
        if not texts:
            return []
        progress = (len(texts) > 100) if show_progress_bar is None else show_progress_bar

        if self.config.embedding_backend == "embeddinggemma2":
            encode_kwargs: dict[str, Any] = {
                "prompt_name": self.config.document_prompt_name,
                "batch_size": self.config.batch_size,
                "normalize_embeddings": True,
                "show_progress_bar": progress,
            }
            if self.config.truncate_dim is not None:
                encode_kwargs["truncate_dim"] = self.config.truncate_dim
            vectors = self.model.encode(texts, **encode_kwargs)
            return vectors.tolist()

        # Baseline multilingual-e5 requires explicit "passage: " prefix.
        vectors = self.model.encode(
            ["passage: " + t for t in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        """Encode a single user query into an L2-normalized vector."""
        q = str(question or "").strip()
        if self.config.embedding_backend == "embeddinggemma2":
            encode_kwargs: dict[str, Any] = {
                "prompt_name": self.config.query_prompt_name,
                "normalize_embeddings": True,
            }
            if self.config.truncate_dim is not None:
                encode_kwargs["truncate_dim"] = self.config.truncate_dim
            vec = self.model.encode([q], **encode_kwargs)[0]
            return vec.tolist()

        vec = self.model.encode(
            ["query: " + q],
            normalize_embeddings=True,
        )[0]
        return vec.tolist()

    def telemetry(self) -> dict[str, Any]:
        """Return honest resource and architecture metadata for trade-off reporting."""
        vram_mb = None
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        return {
            "embedding_model": self.config.embedding_model,
            "backend": self.config.embedding_backend,
            "device": self.device,
            "dtype": str(self.dtype),
            "text_only_encoders": (
                self.config.text_only_encoders
                if self.config.embedding_backend == "embeddinggemma2"
                else None
            ),
            "param_count_millions": round(self.param_count / 1e6, 1),
            "embedding_dim": self.embedding_dim,
            "truncate_dim": self.config.truncate_dim,
            "load_time_s": self.load_time_s,
            "model_vram_mb": self.model_vram_mb,
            "cuda_allocated_mb": vram_mb,
        }
