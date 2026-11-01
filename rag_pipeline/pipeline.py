"""Orchestrates the multi-stage pipeline: sparse + dense -> RRF -> cross-encoder -> LLM."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from rag_pipeline.config import Settings
from rag_pipeline.core.context import assemble_context
from rag_pipeline.core.dense import DenseRetriever, Embedder
from rag_pipeline.core.fusion import reciprocal_rank_fusion
from rag_pipeline.core.generator import GenerationError, Generator
from rag_pipeline.core.reranker import RelevanceScorer, rerank
from rag_pipeline.core.sparse import BM25Retriever
from rag_pipeline.pipelines.chunking import chunk_documents
from rag_pipeline.pipelines.loader import load_documents
from rag_pipeline.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

NO_ANSWER = "I don't know based on the provided documents."
MAX_QUERY_CHARS = 2000


@dataclass
class PipelineResult:
    query: str
    answer: str | None
    sources: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round((time.perf_counter() - start) * 1000, 2)


class RAGPipeline:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        scorer: RelevanceScorer | None,
        generator: Generator | None,
        settings: Settings,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build a pipeline over an empty corpus")
        self.settings = settings
        self.chunks = list(chunks)
        self._sparse = BM25Retriever(self.chunks)
        self._dense = DenseRetriever(self.chunks, embedder)
        self._scorer = scorer
        self._generator = generator

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        """Build the production pipeline: load, chunk, embed and wire the real models."""
        from rag_pipeline.core.dense import SentenceTransformerEmbedder
        from rag_pipeline.core.generator import OllamaGenerator
        from rag_pipeline.core.reranker import CrossEncoderScorer

        documents = load_documents(settings.docs_dir)
        chunks = chunk_documents(
            documents, settings.chunk_max_chars, settings.chunk_overlap_sentences
        )
        logger.info("Indexed %d chunks from %d documents", len(chunks), len(documents))

        generator = None
        if settings.generation_enabled:
            generator = OllamaGenerator(
                settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds
            )
        return cls(
            chunks=chunks,
            embedder=SentenceTransformerEmbedder(settings.embedding_model),
            scorer=CrossEncoderScorer(settings.reranker_model),
            generator=generator,
            settings=settings,
        )

    @staticmethod
    def _validate(query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"Query exceeds {MAX_QUERY_CHARS} characters")
        return query

    def retrieve(self, query: str, top_k: int | None = None) -> PipelineResult:
        query = self._validate(query)
        s = self.settings
        top_k = top_k or s.final_top_k
        timings: dict[str, float] = {}

        with _timed(timings, "sparse"):
            sparse = self._sparse.search(query, s.sparse_top_k)
        with _timed(timings, "dense"):
            dense = self._dense.search(query, s.dense_top_k)
        with _timed(timings, "fusion"):
            fused = reciprocal_rank_fusion([dense, sparse], k=s.rrf_k, top_k=s.rerank_candidates)
        with _timed(timings, "rerank"):
            reranked = rerank(query, fused, self._scorer, top_k, s.rerank_min_score)

        return PipelineResult(
            query=query,
            answer=None,
            sources=reranked.results,
            timings_ms=timings,
            warnings=reranked.warnings,
        )

    def answer(self, query: str, top_k: int | None = None, generate: bool = True) -> PipelineResult:
        start = time.perf_counter()
        result = self.retrieve(query, top_k)

        if not result.sources:
            # Nothing cleared the relevance bar: refuse instead of letting the LLM guess.
            result.answer = NO_ANSWER
            result.warnings.append("no chunks passed the relevance threshold")
        else:
            context = assemble_context(result.sources, self.settings.context_max_tokens)
            result.sources = context.chunks
            if generate and self._generator is not None:
                with _timed(result.timings_ms, "generation"):
                    try:
                        result.answer = self._generator.generate(result.query, context.text)
                    except GenerationError as exc:
                        logger.warning("Generation failed: %s", exc)
                        result.warnings.append(f"generation failed ({exc}); returning sources only")

        result.timings_ms["total"] = round((time.perf_counter() - start) * 1000, 2)
        return result
