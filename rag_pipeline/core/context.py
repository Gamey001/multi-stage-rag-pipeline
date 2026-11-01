"""Packs re-ranked chunks into a token-budgeted, citation-numbered context block."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_pipeline.schemas import ScoredChunk

# ~4 characters per token is a standard heuristic for English text with BPE tokenizers.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class AssembledContext:
    text: str
    chunks: list[ScoredChunk]
    estimated_tokens: int


def _format_entry(index: int, item: ScoredChunk) -> str:
    chunk = item.chunk
    location = f"{chunk.source} > {chunk.section}" if chunk.section else chunk.source
    return f"[{index}] ({location})\n{chunk.text}"


def assemble_context(results: Sequence[ScoredChunk], max_tokens: int) -> AssembledContext:
    """Add chunks in relevance order until the budget is reached; never truncate mid-chunk."""
    entries: list[str] = []
    used: list[ScoredChunk] = []
    total = 0
    for item in results:
        entry = _format_entry(len(used) + 1, item)
        cost = estimate_tokens(entry)
        if total + cost > max_tokens:
            # Skip rather than stop: a shorter, lower-ranked chunk may still fit.
            continue
        entries.append(entry)
        used.append(item)
        total += cost
    return AssembledContext("\n\n".join(entries), used, total)
