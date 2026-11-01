"""Lightweight fakes so the suite runs without downloading transformer models."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from rag_pipeline.core.sparse import tokenize
from rag_pipeline.schemas import Chunk


class HashingEmbedder:
    """Deterministic bag-of-words embedder: shared tokens produce similar vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vectors[row, bucket] += 1.0
        return vectors


class OverlapReranker:
    """Scores by query-token overlap ratio, mimicking a relevance probability in [0, 1]."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        return [
            len(query_tokens & set(tokenize(text))) / max(len(query_tokens), 1) for text in texts
        ]


class FailingReranker:
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        raise RuntimeError("model crashed")


def make_chunk(text: str, idx: int = 0, source: str = "doc.md") -> Chunk:
    return Chunk(id=f"{source}::{idx}", source=source, text=text, position=idx)


@pytest.fixture
def corpus() -> list[Chunk]:
    texts = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "BM25 is a ranking function used by search engines to score keyword relevance.",
        "Reciprocal Rank Fusion merges ranked lists using only rank positions.",
        "Vector embeddings capture semantic meaning for similarity search.",
        "Error code E-4012 means the upload quota was exceeded.",
    ]
    return [make_chunk(text, i) for i, text in enumerate(texts)]
