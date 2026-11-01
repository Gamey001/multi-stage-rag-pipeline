"""Grounded answer generation against a local Ollama server."""

from __future__ import annotations

from typing import Protocol

import httpx

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer using ONLY the numbered context passages. "
    "Cite the passages you use with their numbers, e.g. [1] or [2][3]. "
    "If the context does not contain the answer, reply exactly: "
    '"I don\'t know based on the provided documents." Do not use outside knowledge.'
)


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    def generate(self, question: str, context: str) -> str: ...


def build_user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, question: str, context: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            # Low temperature keeps answers anchored to the retrieved passages.
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
        }
        try:
            response = httpx.post(self._url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except httpx.HTTPError as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationError(f"Unexpected Ollama response: {exc}") from exc

        if not content or not content.strip():
            raise GenerationError("Ollama returned an empty answer")
        return content.strip()
