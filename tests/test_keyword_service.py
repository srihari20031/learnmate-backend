"""Unit tests for BM25 keyword scoring and the per-session BM25 cache."""

from app.services.keyword_service import (
    tokenize,
    BM25,
    get_bm25,
    invalidate_bm25,
    _bm25_cache,
)


# ---- tokenize -------------------------------------------------------------

def test_tokenize_lowercases_and_keeps_underscores():
    # to_thread must survive as ONE token, not split on the underscore.
    assert tokenize("Use asyncio.to_thread NOW") == ["use", "asyncio", "to_thread", "now"]


def test_tokenize_drops_punctuation():
    assert tokenize("hello, world! (test)") == ["hello", "world", "test"]


# ---- BM25 scoring ---------------------------------------------------------

def test_bm25_scores_doc_with_query_term_higher_than_without():
    corpus = [
        tokenize("the event loop and asyncio to_thread"),
        tokenize("completely unrelated cooking recipe"),
    ]
    scores = BM25(corpus).scores("to_thread")
    assert scores[0] > 0          # doc 0 contains the term
    assert scores[1] == 0.0       # doc 1 does not


def test_bm25_rare_term_outweighs_common_term():
    # 'common' appears everywhere (low IDF); 'rare' appears once (high IDF).
    corpus = [
        tokenize("common common rare"),
        tokenize("common common common"),
        tokenize("common word here"),
    ]
    bm25 = BM25(corpus)
    rare_score = bm25.scores("rare")[0]
    common_score = bm25.scores("common")[0]
    assert rare_score > common_score


def test_bm25_empty_corpus_is_safe():
    assert BM25([]).scores("anything") == []


def test_bm25_query_term_absent_everywhere_scores_zero():
    corpus = [tokenize("alpha beta"), tokenize("gamma delta")]
    assert BM25(corpus).scores("missing") == [0.0, 0.0]


# ---- BM25 cache -----------------------------------------------------------

def _chunks(*texts):
    return [{"chunk_id": f"d_{i}", "text": t} for i, t in enumerate(texts)]


def test_cache_hit_returns_same_object():
    _bm25_cache.clear()
    chunks = _chunks("async loop", "bm25 search")
    first, _ = get_bm25("sess-hit", chunks)
    again, _ = get_bm25("sess-hit", chunks)
    assert first is again            # not rebuilt


def test_cache_rebuilds_when_corpus_changes():
    _bm25_cache.clear()
    chunks = _chunks("async loop", "bm25 search")
    first, _ = get_bm25("sess-change", chunks)
    grown, ids = get_bm25("sess-change", chunks + _chunks("reranker"))
    assert grown is not first        # signature changed -> rebuilt
    assert len(ids) == 3


def test_no_session_id_means_no_caching():
    _bm25_cache.clear()
    chunks = _chunks("async loop")
    a, _ = get_bm25(None, chunks)
    b, _ = get_bm25(None, chunks)
    assert a is not b                # never cached without a key
    assert len(_bm25_cache) == 0


def test_invalidate_removes_entry():
    _bm25_cache.clear()
    chunks = _chunks("async loop")
    get_bm25("sess-inv", chunks)
    assert "sess-inv" in _bm25_cache
    invalidate_bm25("sess-inv")
    assert "sess-inv" not in _bm25_cache


def test_returned_ids_align_with_corpus_order():
    _bm25_cache.clear()
    chunks = _chunks("a", "b", "c")
    _, ids = get_bm25("sess-order", chunks)
    assert ids == ["d_0", "d_1", "d_2"]
