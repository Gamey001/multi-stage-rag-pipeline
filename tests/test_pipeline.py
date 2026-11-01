import pytest

from rag_pipeline.config import Settings
from rag_pipeline.core.generator import GenerationError
from rag_pipeline.pipeline import NO_ANSWER, RAGPipeline
from tests.conftest import FailingReranker, HashingEmbedder, OverlapReranker

SETTINGS = Settings(final_top_k=2, rerank_candidates=5, rerank_min_score=0.3)


class EchoGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, context))
        return "grounded answer [1]"


class BrokenGenerator:
    def generate(self, question, context):
        raise GenerationError("ollama offline")


def build(corpus, scorer=None, generator=None):
    return RAGPipeline(corpus, HashingEmbedder(), scorer or OverlapReranker(), generator, SETTINGS)


def test_retrieve_runs_every_stage_and_records_latency(corpus):
    result = build(corpus).retrieve("what does error code E-4012 mean")

    assert result.sources[0].chunk.id == "doc.md::4"
    assert len(result.sources) <= SETTINGS.final_top_k
    assert set(result.timings_ms) == {"sparse", "dense", "fusion", "rerank"}
    assert result.answer is None


def test_answer_passes_numbered_context_to_generator(corpus):
    generator = EchoGenerator()
    result = build(corpus, generator=generator).answer("what is BM25 ranking")

    assert result.answer == "grounded answer [1]"
    question, context = generator.calls[0]
    assert question == "what is BM25 ranking"
    assert context.startswith("[1] (doc.md)")
    assert "generation" in result.timings_ms and "total" in result.timings_ms


def test_refuses_without_calling_llm_when_nothing_is_relevant(corpus):
    generator = EchoGenerator()
    result = build(corpus, generator=generator).answer("kubernetes helm charts")

    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert generator.calls == []
    assert "relevance threshold" in result.warnings[0]


def test_generation_failure_degrades_to_sources_only(corpus):
    result = build(corpus, generator=BrokenGenerator()).answer("what is BM25 ranking")

    assert result.answer is None
    assert result.sources
    assert any("generation failed" in w for w in result.warnings)


def test_reranker_failure_still_returns_fused_results(corpus):
    result = build(corpus, scorer=FailingReranker()).retrieve("BM25 ranking")

    assert result.sources
    assert result.sources[0].stage == "fusion"
    assert any("reranker failed" in w for w in result.warnings)


def test_no_generate_flag_skips_llm(corpus):
    generator = EchoGenerator()
    result = build(corpus, generator=generator).answer("BM25 ranking", generate=False)
    assert result.answer is None and generator.calls == []


@pytest.mark.parametrize("query", ["", "   ", "x" * 2001])
def test_rejects_invalid_queries(corpus, query):
    with pytest.raises(ValueError):
        build(corpus).retrieve(query)


def test_rejects_empty_corpus():
    with pytest.raises(ValueError):
        RAGPipeline([], HashingEmbedder(), None, None, SETTINGS)


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(final_top_k=10, rerank_candidates=5)
    with pytest.raises(ValueError):
        Settings(rrf_k=0)
