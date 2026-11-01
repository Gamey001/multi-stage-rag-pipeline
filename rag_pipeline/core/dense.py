"""Semantic retrieval with a bi-encoder and an in-memory cosine-similarity index."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag_pipeline.schemas import Chunk, ScoredChunk


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Bi-encoder: documents are embedded once offline, queries independently at request time."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class DenseRetriever:
    stage = "dense"

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk")
        self._chunks = list(chunks)
        self._embedder = embedder
        self._matrix = _l2_normalize(embedder.encode([c.searchable_text for c in self._chunks]))
        if self._matrix.shape[0] != len(self._chunks):
            raise ValueError("Embedder returned a different number of vectors than chunks")

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not query.strip() or top_k <= 0:
            return []
        query_vector = _l2_normalize(self._embedder.encode([query]))[0]
        similarities = self._matrix @ query_vector
        k = min(top_k, len(self._chunks))
        # argpartition is O(n); only the k survivors need a full sort.
        candidates = np.argpartition(-similarities, k - 1)[:k]
        order = candidates[np.argsort(-similarities[candidates], kind="stable")]
        return [ScoredChunk(self._chunks[i], float(similarities[i]), self.stage) for i in order]
