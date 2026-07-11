# tests/test_rerank_service.py
#
# The cross-encoder reranker's relevance floor had no test, despite being the fix
# for a real bug: an unrelated AZURE_DEPLOYMENT_GUIDE.pdf was cited as source [5]
# on a question about Redis persistence.
#
# The cause was that `rerank` returned exactly `top_k` chunks, always. It sorted
# by score, sliced to 5, and threw the scores away -- so when only 3 chunks were
# relevant it PADDED the list with the next-best two, however bad. "5th best" was
# enough to be cited.
#
# The fix made top_k a CAP rather than a QUOTA, by dropping anything below a
# relevance floor. Two subtleties make it worth testing:
#
#   1. The cross-encoder emits a LOGIT, not a probability. Strongly positive
#      means the passage answers the query; strongly negative means it does not.
#      ~0.0 is the boundary. (Real scores for the Redis query: redis chunks
#      +7.9 .. +0.3, kafka chunks -8.6, azure chunks -10.3.)
#
#   2. The single best chunk is kept EVEN IF it is below the floor. Otherwise a
#      session that has documents but nothing relevant returns zero context, the
#      grounding guard never gets attached, and the model happily answers from
#      world knowledge instead of saying "the sources don't address this".
#
# We monkeypatch `reranker.predict` so the tests assert on the SELECTION logic,
# not on whatever the ms-marco model happens to think today. Model drift should
# not break these tests; a change to the selection rule should.

from types import SimpleNamespace

import pytest

from app.services import rerank_service
from app.services.rerank_service import RELEVANCE_THRESHOLD, rerank


def _points(*texts):
    # Mimics the Qdrant ScoredPoint shape that retrieve_context_and_sources builds.
    return [
        SimpleNamespace(score=0.0, payload={"text": t, "chunk_id": f"doc_{i}"})
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def scores(monkeypatch):
    """Pin the cross-encoder's output. `scores([...])` sets the logits it returns,
    in the same order as the candidate points passed to rerank()."""

    def _set(values):
        monkeypatch.setattr(rerank_service.reranker, "predict", lambda pairs: list(values))

    return _set


def _ids(points):
    return [p.payload["chunk_id"] for p in points]


def test_the_floor_is_zero_because_the_model_emits_a_logit():
    # Guards the constant itself: a probability-style threshold (0.5) would drop
    # genuinely relevant passages, since logits sit anywhere on the real line.
    assert RELEVANCE_THRESHOLD == 0.0


def test_all_relevant_chunks_are_returned_in_score_order(scores):
    points = _points("redis a", "redis b", "redis c")
    scores([1.0, 7.9, 3.2])  # deliberately NOT in descending order

    result = rerank("how does redis persistence work", points, top_k=5)

    # Reranking reorders: the vector-search order is discarded.
    assert _ids(result) == ["doc_1", "doc_2", "doc_0"]


def test_top_k_is_a_cap_not_a_quota(scores):
    points = _points(*[f"chunk {i}" for i in range(7)])
    scores([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])

    result = rerank("q", points, top_k=5)

    assert len(result) == 5  # capped, even though all 7 clear the floor


def test_chunks_below_the_floor_are_dropped_so_fewer_than_top_k_come_back(scores):
    """The Azure bug, reproduced. Two chunks are relevant; three are not. The old
    code returned all five and cited the garbage."""
    points = _points("redis rdb", "redis aof", "kafka topics", "azure deploy", "resume")
    scores([7.9, 0.3, -8.6, -10.3, -11.4])

    result = rerank("how does redis persistence work", points, top_k=5)

    assert _ids(result) == ["doc_0", "doc_1"]  # not five
    assert all("redis" in p.payload["text"] for p in result)


def test_a_chunk_exactly_at_the_floor_is_kept(scores):
    # `>= min_score`, not `>`.
    points = _points("best", "borderline")
    scores([2.0, 0.0])

    result = rerank("q", points, top_k=5)

    assert _ids(result) == ["doc_0", "doc_1"]


def test_when_nothing_is_relevant_exactly_one_chunk_survives(scores):
    """Keeping the top-1 below the floor is deliberate. With zero chunks the
    grounding guard is never attached to the prompt, and the model answers from
    world knowledge -- e.g. replying "Paris" to "what is the capital of France?"
    against a corpus of Redis notes, instead of declining."""
    points = _points("redis rdb", "redis aof", "redis replication")
    scores([-2.0, -9.0, -5.0])

    result = rerank("what is the capital of France", points, top_k=5)

    assert len(result) == 1
    assert _ids(result) == ["doc_0"]  # the least-bad one, not the first one


def test_no_candidates_returns_empty(scores):
    assert rerank("q", [], top_k=5) == []


def test_points_with_no_text_are_ignored(scores):
    points = _points("real chunk", "")
    scores([5.0])  # only the non-empty candidate is ever scored

    result = rerank("q", points, top_k=5)

    assert _ids(result) == ["doc_0"]


def test_all_points_empty_returns_empty_without_scoring(monkeypatch):
    def _explode(pairs):
        raise AssertionError("the cross-encoder must not run on zero candidates")

    monkeypatch.setattr(rerank_service.reranker, "predict", _explode)
    assert rerank("q", _points("", ""), top_k=5) == []


def test_min_score_is_overridable(scores):
    points = _points("a", "b", "c")
    scores([9.0, 4.0, 1.0])

    # A stricter caller can demand a higher bar; top-1 still survives regardless.
    result = rerank("q", points, top_k=5, min_score=5.0)

    assert _ids(result) == ["doc_0"]
