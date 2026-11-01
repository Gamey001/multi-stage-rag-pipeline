# RAG Failure Modes

## Hallucination From Weak Context

When retrieved passages are only loosely related to the question, language models tend to fill the gap with plausible but unsupported claims. Mitigations are a relevance threshold after re-ranking, a system prompt that restricts the model to the supplied context, numbered citations that make unsupported claims easy to spot, and a low sampling temperature.

## Context Window Overflow

Concatenating every retrieved passage can exceed the model's context window or bury the most relevant passage in the middle, where models attend to it less reliably. Assemble context in relevance order under an explicit token budget and never cut a passage mid-sentence.

## Chunking Errors

Fixed-size character chunking splits sentences and separates a claim from its qualifier. Paragraph-aware chunking keeps paragraphs intact, packs short neighbouring paragraphs together, and splits long paragraphs only at sentence boundaries with a one-sentence overlap so that context carries across the boundary.

## Dependency Outages

The embedding model, the re-ranker and the language model are separate failure points. A resilient pipeline degrades instead of crashing: if the re-ranker fails it returns the fused ranking, and if the language model is unreachable it returns the retrieved sources so the user still gets grounded material.
