import numpy as np
import pytest

from rag_pipeline.core.dense import DenseRetriever
from rag_pipeline.core.sparse import BM25Retriever, tokenize
from tests.conftest import HashingEmbedder


def test_tokenize_preserves_identifiers():
    assert tokenize("Fix E-4012 in user_id v2.1!") == ["fix", "e-4012", "in", "user_id", "v2.1"]


def test_bm25_ranks_exact_keyword_match_first(corpus):
    results = BM25Retriever(corpus).search("what does E-4012 mean", top_k=3)

    assert results[0].chunk.id == "doc.md::4"
    assert all(r.stage == "sparse" for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_bm25_returns_nothing_for_unmatched_or_empty_query(corpus):
    retriever = BM25Retriever(corpus)
    assert retriever.search("zzzz qqqq", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []


def test_dense_retriever_orders_by_cosine_similarity(corpus):
    results = DenseRetriever(corpus, HashingEmbedder()).search("semantic vector embeddings", 2)

    assert len(results) == 2
    assert results[0].chunk.id == "doc.md::3"
    assert results[0].score >= results[1].score
    assert -1.0 <= results[1].score <= 1.0


def test_dense_retriever_caps_top_k_at_corpus_size(corpus):
    results = DenseRetriever(corpus, HashingEmbedder()).search("python", top_k=50)
    assert len(results) == len(corpus)


def test_dense_retriever_rejects_mismatched_embedder(corpus):
    class BadEmbedder:
        def encode(self, texts):
            return np.ones((1, 4))

    with pytest.raises(ValueError):
        DenseRetriever(corpus, BadEmbedder())


def test_retrievers_reject_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever([])
    with pytest.raises(ValueError):
        DenseRetriever([], HashingEmbedder())
