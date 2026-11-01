"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("data/docs")
    chunk_max_chars: int = 800
    chunk_overlap_sentences: int = 1

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    sparse_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_k: int = 3
    rerank_min_score: float = 0.1
    context_max_tokens: int = 1500

    generation_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        positive = {
            "chunk_max_chars": self.chunk_max_chars,
            "sparse_top_k": self.sparse_top_k,
            "dense_top_k": self.dense_top_k,
            "rrf_k": self.rrf_k,
            "rerank_candidates": self.rerank_candidates,
            "final_top_k": self.final_top_k,
            "context_max_tokens": self.context_max_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.chunk_overlap_sentences < 0:
            raise ValueError("chunk_overlap_sentences cannot be negative")
        if self.final_top_k > self.rerank_candidates:
            raise ValueError("final_top_k cannot exceed rerank_candidates")

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        defaults = cls()
        return cls(
            docs_dir=Path(_env_str("RAG_DOCS_DIR", str(defaults.docs_dir))),
            chunk_max_chars=_env_int("RAG_CHUNK_MAX_CHARS", defaults.chunk_max_chars),
            chunk_overlap_sentences=_env_int(
                "RAG_CHUNK_OVERLAP_SENTENCES", defaults.chunk_overlap_sentences
            ),
            embedding_model=_env_str("RAG_EMBEDDING_MODEL", defaults.embedding_model),
            reranker_model=_env_str("RAG_RERANKER_MODEL", defaults.reranker_model),
            sparse_top_k=_env_int("RAG_SPARSE_TOP_K", defaults.sparse_top_k),
            dense_top_k=_env_int("RAG_DENSE_TOP_K", defaults.dense_top_k),
            rrf_k=_env_int("RAG_RRF_K", defaults.rrf_k),
            rerank_candidates=_env_int("RAG_RERANK_CANDIDATES", defaults.rerank_candidates),
            final_top_k=_env_int("RAG_FINAL_TOP_K", defaults.final_top_k),
            rerank_min_score=_env_float("RAG_RERANK_MIN_SCORE", defaults.rerank_min_score),
            context_max_tokens=_env_int("RAG_CONTEXT_MAX_TOKENS", defaults.context_max_tokens),
            generation_enabled=_env_bool("RAG_GENERATION_ENABLED", defaults.generation_enabled),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", defaults.ollama_base_url),
            ollama_model=_env_str("OLLAMA_MODEL", defaults.ollama_model),
            ollama_timeout_seconds=_env_float(
                "OLLAMA_TIMEOUT_SECONDS", defaults.ollama_timeout_seconds
            ),
        )
