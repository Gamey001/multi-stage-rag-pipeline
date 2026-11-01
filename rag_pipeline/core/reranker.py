"""Second-stage precision re-ranking with a cross-encoder.

A cross-encoder reads query and passage together, so attention can catch negations and
constraints ("reset *without* wiping data") that a bi-encoder averages away. It costs one
forward pass per candidate, which is why it only sees the fused shortlist.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag_pipeline.schemas import ScoredChunk

logger = logging.getLogger(__name__)


class RelevanceScorer(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderScorer:
    """Wraps a sentence-transformers CrossEncoder and returns relevance probabilities in [0, 1]."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        # MS MARCO cross-encoders emit raw logits; sigmoid makes thresholds interpretable.
        self._model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())
        self._batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return [float(s) for s in scores]


@dataclass
class RerankResult:
    results: list[ScoredChunk]
    warnings: list[str] = field(default_factory=list)


def rerank(
    query: str,
    candidates: Sequence[ScoredChunk],
    scorer: RelevanceScorer | None,
    top_k: int,
    min_score: float | None = None,
) -> RerankResult:
    """Re-order candidates by cross-encoder relevance and drop those below ``min_score``.

    If the scorer is missing or fails, the fused order is returned unchanged so a model
    outage degrades precision rather than taking the whole service down.
    """
    if not candidates or top_k <= 0:
        return RerankResult([])

    if scorer is None:
        return RerankResult(list(candidates[:top_k]), ["reranker disabled; using fusion order"])

    try:
        scores = list(scorer.score(query, [c.chunk.text for c in candidates]))
        if len(scores) != len(candidates):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(candidates)} inputs")
    except Exception as exc:
        logger.exception("Re-ranking failed; falling back to fusion order")
        return RerankResult(
            list(candidates[:top_k]), [f"reranker failed ({exc}); using fusion order"]
        )

    rescored = sorted(
        (ScoredChunk(c.chunk, s, "rerank") for c, s in zip(candidates, scores)),
        key=lambda r: r.score,
        reverse=True,
    )
    if min_score is not None:
        rescored = [r for r in rescored if r.score >= min_score]
    return RerankResult(rescored[:top_k])
