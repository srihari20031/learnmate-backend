# app/services/keyword_service.py
#
# A small, dependency-free BM25 keyword retriever.
#
# Why this exists: the embedding model matches on *meaning*, but it can miss
# exact terms — function names, identifiers, numbers like "384-dim" or
# "to_thread". BM25 is the classic lexical score: it rewards chunks that contain
# the query's actual words, weighting rare words more and long documents less.
# Pairing it with vector search ("hybrid") covers both failure modes.

import math
import re
from collections import Counter, OrderedDict

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    # Lowercase word/identifier tokens. Keeping underscores means `to_thread`
    # and `worker_threads` survive as single tokens instead of being split.
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """BM25-Okapi ranking over a fixed corpus of pre-tokenized documents."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        # k1 controls term-frequency saturation (how fast extra repeats stop
        # helping); b controls length normalization (how much long docs are
        # penalized). 1.5 / 0.75 are the standard defaults.
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.n = len(corpus_tokens)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(doc) for doc in corpus_tokens]

        # Document frequency: how many docs each term appears in.
        df = Counter()
        for doc in corpus_tokens:
            df.update(set(doc))

        # Inverse document frequency — rare terms score higher. The +0.5
        # smoothing is the standard BM25 form and keeps idf non-negative.
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        # One BM25 score per document for this query (same index order as the
        # corpus passed to __init__).
        scores = [0.0] * self.n
        if not self.avgdl:
            return scores

        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue  # term not in any document
            for i, tf_i in enumerate(self.tf):
                freq = tf_i.get(term, 0)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)

        return scores


# --------------------------------------------------------------------------- #
# Per-session BM25 cache
#
# Building a BM25 index (tokenize every chunk, count term/doc frequencies,
# compute IDF) is CPU work proportional to the whole corpus. The corpus only
# changes when a document is added or re-indexed, but the query changes every
# message — so without caching we rebuild the *same* index on every chat turn.
#
# We key the cache by session and tag it with a cheap signature: the ordered
# tuple of chunk ids. Add/remove/re-index a document and the id set changes, the
# signature changes, and the index is rebuilt automatically — no explicit
# invalidation hook needed. (This relies on get_session_chunks() returning chunks
# in a stable order, which it does by sorting.)
# --------------------------------------------------------------------------- #

_BM25_CACHE_MAX = 128  # cap memory: keep the most recently used N sessions
_bm25_cache: "OrderedDict[str, tuple]" = OrderedDict()


def _corpus_signature(chunks: list[dict]) -> int:
    return hash(tuple(c["chunk_id"] for c in chunks))


def invalidate_bm25(session_id: str | None) -> None:
    # Drop a session's cached index (e.g. when its documents are deleted).
    if session_id:
        _bm25_cache.pop(session_id, None)


def get_bm25(session_id: str | None, chunks: list[dict]):
    """Return (BM25, chunk_ids) for this session's corpus, built once and reused
    while the chunk set is unchanged. chunk_ids align with BM25's score order, so
    callers map score index -> chunk id through it (never through a fresh read,
    whose order could differ from the cached index)."""
    sig = _corpus_signature(chunks)

    cached = _bm25_cache.get(session_id) if session_id else None
    if cached and cached[0] == sig:
        _bm25_cache.move_to_end(session_id)   # mark as recently used
        return cached[1], cached[2]

    chunk_ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25([tokenize(c["text"]) for c in chunks])

    if session_id:
        _bm25_cache[session_id] = (sig, bm25, chunk_ids)
        _bm25_cache.move_to_end(session_id)
        while len(_bm25_cache) > _BM25_CACHE_MAX:
            _bm25_cache.popitem(last=False)    # evict least recently used

    return bm25, chunk_ids
