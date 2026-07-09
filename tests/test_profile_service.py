"""Unit tests for the profile lane's pure logic.

merge_stacks and build_known_stack_preamble are deterministic and touch neither
the LLM nor Mongo, so they unit-test cleanly. extract_stack / save_user_profile /
resolve_known_stack are async I/O and belong in an integration layer.
"""

from app.services.profile_service import merge_stacks
from app.services.claude_service import build_known_stack_preamble


# ---- merge_stacks ---------------------------------------------------------

def test_merge_into_empty_existing_returns_incoming():
    assert merge_stacks("", "Python, React") == "Python, React"
    assert merge_stacks(None, "Python") == "Python"


def test_merge_with_empty_incoming_keeps_existing():
    assert merge_stacks("Python, React", "") == "Python, React"
    assert merge_stacks("Python", None) == "Python"


def test_both_empty_returns_empty_string():
    assert merge_stacks(None, None) == ""
    assert merge_stacks("", "") == ""


def test_union_preserves_order_and_appends_new():
    # existing order is preserved; only genuinely new items are appended.
    assert merge_stacks("Python, FastAPI", "Redis, Docker") == "Python, FastAPI, Redis, Docker"


def test_case_insensitive_dedup():
    # "python" already present (as "Python") must not be re-added.
    assert merge_stacks("Python, React", "python, Redis") == "Python, React, Redis"


def test_no_additions_returns_existing_unchanged():
    assert merge_stacks("Python, React", "react, PYTHON") == "Python, React"


def test_whitespace_around_items_is_normalized():
    assert merge_stacks("Python ,  React", "  Redis ") == "Python, React, Redis"


def test_incoming_internal_dedup():
    # duplicates within the incoming list collapse too.
    assert merge_stacks("", "Go, go, GO, Rust") == "Go, Rust"


# ---- build_known_stack_preamble ------------------------------------------

def test_preamble_includes_the_stack():
    p = build_known_stack_preamble("Python, FastAPI, MongoDB")
    assert "Python, FastAPI, MongoDB" in p


def test_preamble_tells_model_not_to_re_ask():
    p = build_known_stack_preamble("Python").lower()
    assert "do not make them restate" in p
    # Stays open to skills the profile doesn't capture (the user's requirement).
    assert "else" in p


def test_preamble_never_calls_the_profile_an_uploaded_document():
    # The preamble used to say the stack came "via an uploaded resume", which led the
    # model to recite the profile when asked "what does my uploaded document say?" in a
    # chat with no documents. It must now disclaim that explicitly and never say "resume".
    p = build_known_stack_preamble("Python").lower()
    assert "not a document they uploaded" in p
    assert "resume" not in p
