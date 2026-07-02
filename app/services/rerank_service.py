# app/services/rerank_service.py

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# A bi-encoder (the embedding model) scores the query and each chunk *separately*,
# so it never sees them together — fast, but approximate. A cross-encoder feeds the
# (query, chunk) pair through the model jointly and scores their actual relevance,
# which is far more accurate but too slow to run over the whole collection. So we
# use the bi-encoder to fetch a candidate pool, then the cross-encoder to re-order it.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, points: list, top_k: int = 5) -> list:
    # `points` are Qdrant ScoredPoint objects from search_embeddings(). Returns the
    # same point objects, re-ordered by cross-encoder relevance, truncated to top_k.
    candidates = [(p, p.payload.get("text", "")) for p in points]
    candidates = [(p, text) for p, text in candidates if text]

    if not candidates:
        return []

    pairs = [(query, text) for _, text in candidates]
    scores = reranker.predict(pairs)

    # `candidates` is still in the vector-search order Qdrant returned (best-first),
    # so its index is each point's original rank before reranking.
    ordered = list(zip((p for p, _ in candidates), scores))
    ranked = sorted(ordered, key=lambda item: float(item[1]), reverse=True)

    _log_movement(query, ordered, ranked, top_k)

    return [point for point, _ in ranked[:top_k]]


def _log_movement(query: str, ordered: list, ranked: list, top_k: int) -> None:
    # Skip the work entirely if this log level is muted.
    if not logger.isEnabledFor(logging.INFO):
        return

    def _key(point):
        return point.payload.get("chunk_id", point.payload.get("chunk_index"))

    original_rank = {_key(point): i for i, (point, _) in enumerate(ordered)}

    logger.info("Rerank %d candidates -> top %d for query: %r",
                len(ordered), top_k, query[:80])

    for new_pos, (point, ce_score) in enumerate(ranked[:top_k]):
        was = original_rank.get(_key(point))
        moved = "=" if was == new_pos else f"{was}->{new_pos}"
        preview = (point.payload.get("text") or "")[:60].replace("\n", " ")
        logger.info(
            "  rank %d (vec #%s, %s)  vec=%.3f  ce=%.3f | %s",
            new_pos, was, moved, float(point.score or 0.0), float(ce_score), preview,
        )
