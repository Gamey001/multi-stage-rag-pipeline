import pytest
from fastapi.testclient import TestClient

from rag_pipeline.api.app import create_app
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline
from tests.conftest import HashingEmbedder, OverlapReranker


class StubGenerator:
    def generate(self, question, context):
        return "BM25 scores keyword relevance [1]."


@pytest.fixture
def client(corpus):
    settings = Settings(final_top_k=2, rerank_candidates=5, rerank_min_score=0.3)

    def factory():
        return RAGPipeline(corpus, HashingEmbedder(), OverlapReranker(), StubGenerator(), settings)

    with TestClient(create_app(factory)) as test_client:
        yield test_client


def test_health_reports_index_size(client, corpus):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == len(corpus)


def test_query_returns_answer_sources_and_timings(client):
    response = client.post("/query", json={"query": "what is BM25 ranking"})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == "BM25 scores keyword relevance [1]."
    assert body["sources"][0]["source"] == "doc.md"
    assert "total" in body["timings_ms"]


def test_retrieve_skips_generation(client):
    body = client.post("/retrieve", json={"query": "BM25 ranking", "top_k": 1}).json()
    assert body["answer"] is None
    assert len(body["sources"]) == 1


@pytest.mark.parametrize(
    "payload", [{"query": ""}, {"query": "x" * 2001}, {"query": "ok", "top_k": 0}, {}]
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_whitespace_query_is_rejected(client):
    assert client.post("/query", json={"query": "   "}).status_code == 422
