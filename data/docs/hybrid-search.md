# Hybrid Search

Hybrid search combines lexical retrieval and semantic retrieval so that each method covers the other's blind spots. Lexical retrieval, usually BM25, matches the exact terms in a query. Semantic retrieval compares dense vector embeddings and matches meaning even when the wording differs.

## Why Vector Search Alone Fails

Dense embeddings compress a passage into a single vector. Rare tokens such as error codes, product SKUs, function names and ticket IDs contribute little to that vector, so a query for "E-4012" can return passages about generic upload errors instead of the one passage that mentions E-4012. BM25 scores that exact token highly because it is rare across the corpus.

## Why BM25 Alone Fails

BM25 has no notion of synonyms or paraphrase. A user asking "how do I make my API faster" will not match a passage titled "Reducing endpoint latency" unless the words overlap. Dense retrieval closes this vocabulary gap.

## BM25 Scoring

BM25 ranks a passage by summing, for each query term, the inverse document frequency of the term multiplied by a saturated term frequency. The parameter k1 controls how quickly repeated occurrences stop adding score, and b controls how strongly long passages are penalised. Scores are unbounded and depend on corpus statistics, so they cannot be compared directly with cosine similarity.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges ranked lists using only rank positions. Each retriever contributes 1 / (k + rank) for every passage it returns, and the contributions are summed. The constant k is typically 60. Because raw scores are ignored, RRF avoids the score-scale mismatch between BM25 and cosine similarity and needs no per-corpus tuning. Passages that appear high in both lists rise to the top.
