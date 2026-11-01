"""Loads raw text documents from disk."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_pipeline.schemas import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".rst"})


def load_documents(root: Path | str) -> list[Document]:
    """Recursively load supported text files under ``root``, sorted for deterministic indexing."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Document directory not found: {root}")

    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    documents: list[Document] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
            continue
        if not text.strip():
            logger.warning("Skipping empty file %s", path)
            continue
        source = path.name if root.is_file() else str(path.relative_to(root))
        documents.append(Document(source=source, text=text))

    if not documents:
        raise ValueError(f"No supported documents found in {root}")
    logger.info("Loaded %d documents from %s", len(documents), root)
    return documents
