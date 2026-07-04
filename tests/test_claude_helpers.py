"""Unit tests for the grounding/injection guard and citation marking.

These import claude_service, which constructs the (offline) Groq client but does
NOT load any embedding/rerank model at import time, so they stay fast.
"""

from app.services.claude_service import (
    build_context_guard,
    mark_cited_sources,
    MAX_CONTEXT_CHARS,
)


# ---- build_context_guard --------------------------------------------------

def test_guard_contains_grounding_instruction():
    guard = build_context_guard("[1] some retrieved text")
    lowered = guard.lower()
    assert "only" in lowered                     # answer ONLY from sources
    assert "outside knowledge" in lowered        # anti-hallucination clause


def test_guard_contains_injection_instruction():
    guard = build_context_guard("[1] some retrieved text")
    assert "UNTRUSTED" in guard
    assert "NEVER follow" in guard


def test_guard_wraps_context_in_matching_fence():
    body = "[1] the actual chunk body"
    guard = build_context_guard(body)
    # Every ctx-token in the guard (in the instruction prose AND the fence tags)
    # must be the SAME random value, and the body must sit inside a matching
    # open/close fence.
    import re
    tokens = set(re.findall(r"</?ctx-([0-9a-f]+)>", guard))
    assert len(tokens) == 1                        # one consistent token
    token = tokens.pop()
    assert f"<ctx-{token}>\n{body}\n</ctx-{token}>" in guard


def test_guard_fence_is_random_per_call():
    import re
    t1 = re.search(r"<ctx-([0-9a-f]+)>", build_context_guard("x")).group(1)
    t2 = re.search(r"<ctx-([0-9a-f]+)>", build_context_guard("x")).group(1)
    assert t1 != t2                               # unforgeable per request


def test_guard_truncates_oversized_context():
    huge = "z" * (MAX_CONTEXT_CHARS + 5000)
    guard = build_context_guard(huge)
    assert huge[:MAX_CONTEXT_CHARS] in guard
    assert "z" * (MAX_CONTEXT_CHARS + 1) not in guard   # nothing past the cap


# ---- mark_cited_sources ---------------------------------------------------

def _sources(n):
    return [{"id": i, "cited": False} for i in range(1, n + 1)]


def test_in_range_citations_are_flagged():
    sources = _sources(3)
    mark_cited_sources("Use to_thread [1]. BM25 is lexical [2][3].", sources)
    assert {s["id"]: s["cited"] for s in sources} == {1: True, 2: True, 3: True}


def test_out_of_range_marker_is_ignored():
    # arr[9] is code, not a citation — there is no source #9.
    sources = _sources(3)
    mark_cited_sources("See arr[9] in the loop [1].", sources)
    flags = {s["id"]: s["cited"] for s in sources}
    assert flags == {1: True, 2: False, 3: False}


def test_uncited_sources_stay_false():
    sources = _sources(2)
    mark_cited_sources("An answer with no citations at all.", sources)
    assert all(s["cited"] is False for s in sources)


def test_empty_sources_is_safe():
    assert mark_cited_sources("anything [1]", []) == []


def test_none_reply_is_safe():
    sources = _sources(2)
    mark_cited_sources(None, sources)
    assert all(s["cited"] is False for s in sources)
