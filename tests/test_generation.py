import httpx
import pytest

from rag_pipeline.core import generator as generator_module
from rag_pipeline.core.context import assemble_context, estimate_tokens
from rag_pipeline.core.generator import GenerationError, OllamaGenerator
from rag_pipeline.schemas import Chunk, ScoredChunk


def scored(text, idx, section=""):
    chunk = Chunk(id=f"d.md::{idx}", source="d.md", text=text, position=idx, section=section)
    return ScoredChunk(chunk, 0.9, "rerank")


def test_context_numbers_entries_and_includes_location():
    ctx = assemble_context([scored("Alpha.", 0, "Intro"), scored("Beta.", 1)], max_tokens=100)

    assert ctx.text == "[1] (d.md > Intro)\nAlpha.\n\n[2] (d.md)\nBeta."
    assert len(ctx.chunks) == 2
    assert ctx.estimated_tokens > 0


def test_context_respects_budget_and_skips_oversized_chunks():
    big = scored("x" * 400, 0)
    small = scored("short", 1)
    ctx = assemble_context([big, small], max_tokens=20)

    assert ctx.chunks == [small]
    assert ctx.text.startswith("[1]")
    assert ctx.estimated_tokens <= 20


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") == 1


def _mock_post(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def fake_post(url, json, timeout):
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(generator_module.httpx, "post", fake_post)


def test_ollama_generator_sends_grounded_prompt(monkeypatch):
    captured = {}

    def handler(request):
        captured["body"] = request.read().decode()
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"message": {"content": " Answer [1] "}})

    _mock_post(monkeypatch, handler)
    answer = OllamaGenerator("http://ollama:11434/", "llama3.2").generate("Q?", "[1] ctx")

    assert answer == "Answer [1]"
    assert captured["url"] == "http://ollama:11434/api/chat"
    assert "ONLY the numbered context" in captured["body"]
    assert "[1] ctx" in captured["body"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, json={"message": {"content": "   "}}),
    ],
)
def test_ollama_generator_raises_generation_error(monkeypatch, response):
    _mock_post(monkeypatch, lambda request: response)
    with pytest.raises(GenerationError):
        OllamaGenerator("http://ollama:11434", "llama3.2").generate("Q?", "ctx")


def test_ollama_generator_wraps_connection_errors(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _mock_post(monkeypatch, handler)
    with pytest.raises(GenerationError, match="Ollama request failed"):
        OllamaGenerator("http://ollama:11434", "llama3.2").generate("Q?", "ctx")
