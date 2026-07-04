"""Unit tests for RRF fusion and hybrid_rank."""

from app.services.hybrid_service import reciprocal_rank_fusion, hybrid_rank
from app.services.keyword_service import _bm25_cache


# ---- reciprocal_rank_fusion ----------------------------------------------

def test_rrf_item_in_both_lists_beats_item_in_one():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "x"]
    fused = reciprocal_rank_fusion([dense, sparse])
    # 'a' is high in both; 'c' is 3rd in dense but 1st in sparse; both beat
    # singletons like 'b' and 'x'.
    assert fused[0] == "a"
    assert fused.index("a") < fused.index("b")
    assert fused.index("c") < fused.index("x")


def test_rrf_single_list_preserves_order():
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rank_beats_raw_score_scale():
    # Fusion must use rank position only. Even if one retriever's "scores" were
    # on a wild scale, RRF only sees positions — so an item ranked #1 in the
    # short list and #1 in the long list wins.
    a = ["win", "z1", "z2", "z3", "z4"]
    b = ["win", "y1"]
    assert reciprocal_rank_fusion([a, b])[0] == "win"


# ---- hybrid_rank ----------------------------------------------------------

def _chunks(*texts):
    return [{"chunk_id": f"c{i}", "text": t} for i, t in enumerate(texts)]


def test_hybrid_rank_surfaces_keyword_match_dense_missed():
    _bm25_cache.clear()
    chunks = _chunks(
        "the async event loop offloads work",   # c0
        "reranking with a cross encoder",        # c1
        "bm25 lexical keyword search rocks",     # c2  <- the keyword answer
    )
    # Dense retriever ranked the keyword chunk LAST (missed it).
    dense_ids = ["c0", "c1", "c2"]
    fused = hybrid_rank("bm25 keyword", chunks, dense_ids, pool_size=3, session_id=None)
    # BM25 pulls c2 up; fusion should rank it above c1 which neither favored.
    assert "c2" in fused
    assert fused.index("c2") < fused.index("c1")


def test_hybrid_rank_respects_pool_size():
    _bm25_cache.clear()
    chunks = _chunks("a b", "c d", "e f", "g h")
    fused = hybrid_rank("a", chunks, ["c0", "c1", "c2", "c3"], pool_size=2, session_id=None)
    assert len(fused) == 2


def test_hybrid_rank_ids_are_valid_chunk_ids():
    _bm25_cache.clear()
    chunks = _chunks("alpha", "beta", "gamma")
    fused = hybrid_rank("alpha", chunks, ["c0", "c1", "c2"], pool_size=3, session_id=None)
    assert set(fused) <= {"c0", "c1", "c2"}
