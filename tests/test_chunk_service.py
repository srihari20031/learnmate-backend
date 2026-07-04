"""Unit tests for token-based chunking (chunk_service).

These load the MiniLM tokenizer once, so this module is slightly heavier than
the pure-logic tests — but chunk sizing is the fix that started the whole RAG
cleanup, so it's worth guarding.
"""

import pytest

from app.services.chunk_service import chunk_text, tokenizer

TOKEN_LIMIT = 256


def _tok_len(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def test_every_chunk_within_embedding_token_limit():
    # The whole point of the 256 fix: no chunk may exceed the model's window.
    text = "word " * 2000  # ~2000 tokens, well over one chunk
    for chunk in chunk_text(text):
        assert _tok_len(chunk) <= TOKEN_LIMIT


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_short_text_below_min_is_dropped():
    # Chunks under 50 tokens are skipped, so a tiny input yields nothing.
    assert chunk_text("just a few words") == []


def test_long_text_produces_multiple_chunks():
    chunks = chunk_text("word " * 1000)
    assert len(chunks) > 1


def test_chunks_are_nonempty_strings():
    chunks = chunk_text("sentence number one. " * 200)
    assert chunks
    assert all(isinstance(c, str) and c.strip() for c in chunks)
