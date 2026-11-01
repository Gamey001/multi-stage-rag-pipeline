"""Paragraph- and sentence-aware chunking.

Chunks never split a sentence. Short paragraphs in the same section are packed together,
long paragraphs are split on sentence boundaries with a small sentence overlap, and
markdown headings are carried as section metadata instead of becoming standalone chunks.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from rag_pipeline.schemas import Chunk, Document

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
# Split after terminal punctuation followed by whitespace and an uppercase letter, digit or quote.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [s for s in _SENTENCE_SPLIT.split(normalized) if s]


def _iter_paragraphs(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(section, paragraph)`` pairs, tracking the most recent markdown heading."""
    section = ""
    for block in _PARAGRAPH_SPLIT.split(text):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        body: list[str] = []
        for line in lines:
            match = _HEADING.match(line.strip())
            if match:
                if body:
                    yield section, "\n".join(body)
                    body = []
                section = match.group(1).strip()
            else:
                body.append(line)
        if body:
            yield section, "\n".join(body)


def _split_long_paragraph(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    window: list[str] = []
    for sentence in split_sentences(paragraph):
        if window and len(" ".join([*window, sentence])) > max_chars:
            pieces.append(" ".join(window))
            window = window[-overlap:] if overlap else []
            if window and len(" ".join([*window, sentence])) > max_chars:
                window = []
        window.append(sentence)
    if window:
        pieces.append(" ".join(window))
    return pieces


def chunk_document(
    document: Document, max_chars: int = 800, overlap_sentences: int = 1
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_section = ""

    def emit(text: str, section: str) -> None:
        position = len(chunks)
        chunks.append(
            Chunk(
                id=f"{document.source}::{position}",
                source=document.source,
                text=text,
                position=position,
                section=section,
            )
        )

    def flush() -> None:
        if buffer:
            emit("\n\n".join(buffer), buffer_section)
            buffer.clear()

    for section, paragraph in _iter_paragraphs(document.text):
        if section != buffer_section:
            flush()
            buffer_section = section

        if buffer and len("\n\n".join([*buffer, paragraph])) <= max_chars:
            buffer.append(paragraph)
        elif len(paragraph) <= max_chars:
            flush()
            buffer.append(paragraph)
        else:
            flush()
            for piece in _split_long_paragraph(paragraph, max_chars, overlap_sentences):
                emit(piece, section)

    flush()
    return chunks


def chunk_documents(
    documents: list[Document], max_chars: int = 800, overlap_sentences: int = 1
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, max_chars, overlap_sentences)
    ]
