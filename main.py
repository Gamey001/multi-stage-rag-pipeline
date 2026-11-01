"""Command-line entry point: python main.py "your question" [--no-generate]."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import PipelineResult, RAGPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the multi-stage RAG pipeline.")
    parser.add_argument("query", help="Question to ask the knowledge base")
    parser.add_argument("--docs", type=Path, help="Directory of documents to index")
    parser.add_argument("--top-k", type=int, help="Number of chunks to return")
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip the LLM and print retrieved chunks only"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def render(result: PipelineResult) -> str:
    lines = [f"Query: {result.query}", ""]
    if result.answer:
        lines += ["Answer:", result.answer, ""]
    lines.append("Sources:")
    for i, item in enumerate(result.sources, start=1):
        chunk = item.chunk
        location = f"{chunk.source} > {chunk.section}" if chunk.section else chunk.source
        preview = " ".join(chunk.text.split())[:160]
        lines.append(f"  [{i}] {item.stage}={item.score:.4f}  {location}\n      {preview}")
    if not result.sources:
        lines.append("  (none)")
    timings = ", ".join(f"{stage}={ms}ms" for stage, ms in result.timings_ms.items())
    lines += ["", f"Latency: {timings}"]
    lines += [f"Warning: {w}" for w in result.warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    if args.docs:
        settings = replace(settings, docs_dir=args.docs)
    if args.no_generate:
        settings = replace(settings, generation_enabled=False)

    try:
        pipeline = RAGPipeline.from_settings(settings)
        result = pipeline.answer(args.query, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
