"""Core data types shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    position: int
    section: str = ""

    @property
    def searchable_text(self) -> str:
        """Text used for indexing; the section heading adds context to short chunks."""
        return f"{self.section}\n{self.text}" if self.section else self.text


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    stage: str
