# Re-ranking With Cross-Encoders

## Bi-Encoders Versus Cross-Encoders

A bi-encoder embeds the query and each passage independently. Passage vectors are computed once at indexing time, so search over millions of passages takes milliseconds. The cost is precision: the model never sees the query and passage together, so it cannot model how a word in the query modifies a word in the passage.

A cross-encoder feeds the query and passage into the transformer as one sequence. Attention flows across both texts, which yields a far more accurate relevance score. It must run once per query-passage pair at request time, so it is too slow to apply to a whole corpus.

## Multi-Stage Retrieval

Production systems use the bi-encoder and BM25 as a cheap, high-recall first stage that narrows the corpus to roughly twenty candidates. The cross-encoder then re-scores only those candidates and keeps the top three to five for the language model. This spends expensive computation only where it changes the outcome.

## Relevance Thresholds

Re-ranker scores can be converted to probabilities with a sigmoid. Dropping passages below a minimum probability prevents irrelevant context from reaching the language model. When no passage clears the threshold, the safest behaviour is to answer that the documents do not contain the information rather than letting the model guess.
