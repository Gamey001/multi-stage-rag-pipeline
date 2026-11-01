import pytest

from rag_pipeline.core.fusion import reciprocal_rank_fusion
from rag_pipeline.schemas import ScoredChunk
from tests.conftest import make_chunk

A, B, C, D = (make_chunk(t, i) for i, t in enumerate("abcd"))


def ranked(*chunks, stage="dense"):
    # Raw scores are deliberately on wildly different scales; RRF must ignore them.
    return [ScoredChunk(c, 1000.0 - i, stage) for i, c in enumerate(chunks)]


def test_rrf_scores_match_formula():
    fused = reciprocal_rank_fusion([ranked(A, B), ranked(B, C)], k=60)
    scores = {r.chunk.id: r.score for r in fused}

    assert scores[B.id] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[A.id] == pytest.approx(1 / 61)
    assert scores[C.id] == pytest.approx(1 / 62)
    assert [r.chunk.id for r in fused] == [B.id, A.id, C.id]
    assert all(r.stage == "fusion" for r in fused)


def test_consensus_beats_single_top_rank():
    dense = ranked(A, B, C)
    sparse = ranked(D, C, B)
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].chunk.id in {B.id, C.id}


def test_ignores_raw_score_magnitudes():
    tiny = [ScoredChunk(A, 0.001, "dense"), ScoredChunk(B, 0.0001, "dense")]
    huge = [ScoredChunk(B, 9e9, "sparse"), ScoredChunk(A, 1.0, "sparse")]
    fused = reciprocal_rank_fusion([tiny, huge])
    assert fused[0].score == pytest.approx(fused[1].score)


def test_duplicates_within_one_list_vote_once():
    fused = reciprocal_rank_fusion([[*ranked(A), *ranked(A)]])
    assert fused[0].score == pytest.approx(1 / 61)


def test_top_k_and_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert len(reciprocal_rank_fusion([ranked(A, B, C)], top_k=2)) == 2


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranked(A)], k=0)
