"""HTTP interface. Run with: uvicorn rag_pipeline.api.app:create_app --factory"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag_pipeline import __version__
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import MAX_QUERY_CHARS, PipelineResult, RAGPipeline


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=20)
    generate: bool = True


class SourceOut(BaseModel):
    id: str
    source: str
    section: str
    text: str
    score: float
    stage: str


class QueryResponse(BaseModel):
    query: str
    answer: str | None
    sources: list[SourceOut]
    timings_ms: dict[str, float]
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks_indexed: int


def _to_response(result: PipelineResult) -> QueryResponse:
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            SourceOut(
                id=item.chunk.id,
                source=item.chunk.source,
                section=item.chunk.section,
                text=item.chunk.text,
                score=item.score,
                stage=item.stage,
            )
            for item in result.sources
        ],
        timings_ms=result.timings_ms,
        warnings=result.warnings,
    )


def create_app(pipeline_factory: Callable[[], RAGPipeline] | None = None) -> FastAPI:
    factory = pipeline_factory or (lambda: RAGPipeline.from_settings(Settings.from_env()))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load models and build indexes once at startup, not per request.
        app.state.pipeline = factory()
        yield

    app = FastAPI(title="Multi-Stage RAG Pipeline", version=__version__, lifespan=lifespan)

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pipeline = get_pipeline(request)
        return HealthResponse(status="ok", version=__version__, chunks_indexed=len(pipeline.chunks))

    # Sync handlers run in FastAPI's threadpool, so CPU-bound inference won't block the loop.
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request) -> QueryResponse:
        try:
            result = get_pipeline(request).answer(body.query, body.top_k, body.generate)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _to_response(result)

    @app.post("/retrieve", response_model=QueryResponse)
    def retrieve(body: QueryRequest, request: Request) -> QueryResponse:
        try:
            result = get_pipeline(request).retrieve(body.query, body.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _to_response(result)

    return app
