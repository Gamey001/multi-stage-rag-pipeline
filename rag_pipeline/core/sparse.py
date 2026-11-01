"""Lexical retrieval with BM25 (Okapi)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from rag_pipeline.schemas import Chunk, ScoredChunk

# Keeps identifiers such as "http-404", "user_id" and "v2.1" as single tokens.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Exact-term matcher that catches error codes, IDs and rare keywords embeddings blur."""

    stage = "sparse"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self._chunks = list(chunks)
        self._index = BM25Okapi([tokenize(c.searchable_text) for c in self._chunks])

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if not tokens or top_k <= 0:
            return []
        scores = self._index.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            ScoredChunk(self._chunks[i], float(scores[i]), self.stage)
            for i in order
            if scores[i] > 0
        ]
