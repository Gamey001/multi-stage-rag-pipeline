"""Reciprocal Rank Fusion (Cormack et al., 2009).

BM25 scores are unbounded while cosine similarity lives in [-1, 1], so normalizing raw
scores is brittle. RRF uses only rank positions:

    RRF(d) = sum over retrievers m of 1 / (k + rank_m(d))

A larger ``k`` flattens the advantage of top ranks; 60 is the value from the original paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_pipeline.schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    k: int = 60,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            # A retriever should only vote once per chunk, even if it returned duplicates.
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.setdefault(chunk_id, result.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # sorted() is stable, so ties keep first-seen order and output stays deterministic.
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [ScoredChunk(chunks[chunk_id], score, "fusion") for chunk_id, score in ranked]
