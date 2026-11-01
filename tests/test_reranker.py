from rag_pipeline.core.reranker import rerank
from rag_pipeline.schemas import ScoredChunk
from tests.conftest import FailingReranker, OverlapReranker, make_chunk


def fused(*texts):
    return [ScoredChunk(make_chunk(t, i), 1.0 / (61 + i), "fusion") for i, t in enumerate(texts)]


def test_reorders_candidates_by_scorer_relevance():
    candidates = fused("unrelated text here", "reset laptop keep documents", "reset laptop")
    result = rerank("reset laptop keep documents", candidates, OverlapReranker(), top_k=2)

    assert [r.chunk.text for r in result.results] == [
        "reset laptop keep documents",
        "reset laptop",
    ]
    assert all(r.stage == "rerank" for r in result.results)
    assert result.warnings == []


def test_min_score_filters_irrelevant_chunks():
    candidates = fused("totally unrelated", "reset laptop")
    result = rerank("reset laptop", candidates, OverlapReranker(), top_k=5, min_score=0.5)
    assert [r.chunk.text for r in result.results] == ["reset laptop"]


def test_falls_back_to_fusion_order_when_scorer_fails():
    candidates = fused("a", "b", "c")
    result = rerank("q", candidates, FailingReranker(), top_k=2)

    assert result.results == candidates[:2]
    assert "reranker failed" in result.warnings[0]


def test_falls_back_when_scorer_returns_wrong_length():
    class ShortScorer:
        def score(self, query, texts):
            return [1.0]

    result = rerank("q", fused("a", "b"), ShortScorer(), top_k=2)
    assert len(result.results) == 2
    assert result.warnings


def test_disabled_scorer_and_empty_candidates():
    assert rerank("q", [], OverlapReranker(), top_k=3).results == []
    result = rerank("q", fused("a", "b"), None, top_k=1)
    assert len(result.results) == 1
    assert "disabled" in result.warnings[0]
