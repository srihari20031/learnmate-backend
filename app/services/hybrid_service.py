# app/services/hybrid_service.py
#
# Hybrid retrieval = combine the dense (vector) ranking with the sparse (BM25)
# ranking into one candidate list, then let the reranker do the fine sorting.
#
# We fuse with Reciprocal Rank Fusion (RRF). RRF ignores the raw scores (which
# live on totally different scales — cosine ~0-1 vs unbounded BM25) and uses only
# each item's *position* in each list. A chunk near the top of either list gets a
# high fused score; a chunk near the top of *both* wins. It's simple, scale-free,
# and the standard way to merge heterogeneous rankers.

from app.services.keyword_service import get_bm25


def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list:
    # rankings: several ranked id-lists (best-first). Returns one fused id-list.
    # Each list contributes 1 / (k + rank) per item; k (60 by convention) damps
    # the influence of the very top ranks so a single list can't dominate.
    scores: dict = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


def hybrid_rank(
    query: str,
    chunks: list[dict],
    dense_ids: list,
    pool_size: int,
    k: int = 60,
    session_id: str | None = None,
) -> list:
    """Fuse a dense id ranking with a BM25 ranking over `chunks`.

    chunks     : [{"chunk_id": ..., "text": ...}] — the full scope to search.
    dense_ids  : chunk ids from vector search, already best-first.
    session_id : cache key; the BM25 index is reused across messages until the
                 session's chunk set changes (None = build fresh, no caching).
    returns    : fused chunk ids, best-first, truncated to pool_size.
    """
    bm25, chunk_ids = get_bm25(session_id, chunks)
    sparse_scores = bm25.scores(query)

    # chunk_ids comes from the (possibly cached) index, so it aligns with the
    # score order — don't index back into `chunks` here.
    sparse_order = sorted(
        range(len(chunk_ids)), key=lambda i: sparse_scores[i], reverse=True
    )
    sparse_ids = [chunk_ids[i] for i in sparse_order[:pool_size]]

    fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=k)
    return fused[:pool_size]
