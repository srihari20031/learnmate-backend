"""
Retrieval evaluation harness.

Run from the repo root:

    python -m evals.run_eval

WHAT THIS MEASURES
------------------
For a set of known questions, does our retrieval put the *correct* chunk near
the top? We run four configurations and print their scores side by side:

    1. "vector only"   — dense embedding similarity search alone
    2. "hybrid only"   — dense + BM25 keyword, fused with RRF (no reranker)
    3. "vector+rerank" — dense recall, then the cross-encoder reorders
    4. "hybrid+rerank" — the full production pipeline: hybrid recall + rerank

Reading the table:
  - 1 vs 2 isolates what *hybrid* adds at the recall stage.
  - 1 vs 3 (and 2 vs 4) isolates what the *reranker* adds.
  - 4 vs 3 is the headline: does the new hybrid pipeline beat the old baseline?

WHY THERE'S NO QDRANT OR MONGO HERE
-----------------------------------
Qdrant's job is cosine nearest-neighbour search. We reproduce exactly that with
a few lines of numpy over the same vectors, so the eval needs no running
database and never touches your real data. The important parts are still the
REAL ones: the production embedding model, the production BM25 + RRF fusion, and
the production rerank() function. So we measure true ranking quality, just
without the DB plumbing.
"""

import logging
from types import SimpleNamespace

import numpy as np

# Reuse the exact production components so the eval reflects real behaviour.
from app.services.embedding import generate_embeddings, generate_embeddings_batch
from app.services.rerank_service import rerank
from app.services.keyword_service import BM25, tokenize
from app.services.hybrid_service import reciprocal_rank_fusion

from evals.dataset import CORPUS, QUESTIONS

# The reranker logs every reordering at INFO; silence it so our table stays
# readable. Flip this to logging.INFO if you want to watch the chunks move.
logging.getLogger("app.services.rerank_service").setLevel(logging.WARNING)

# How many candidates the recall stage hands to the reranker. In production this
# is 20; here it's capped at the corpus size because our toy corpus is tiny.
CANDIDATE_POOL = min(20, len(CORPUS))
FINAL_TOP_K = 5

CONFIGS = ("vector only", "hybrid only", "vector+rerank", "hybrid+rerank")


def cosine_rank(query_vec, corpus_matrix):
    """Return (corpus_index, score) pairs sorted best-first by cosine similarity.

    Cosine similarity is just the dot product of L2-normalised vectors. This is
    exactly what Qdrant computes with Distance.COSINE — we're doing it locally.
    """
    q = np.asarray(query_vec, dtype=np.float32)
    M = np.asarray(corpus_matrix, dtype=np.float32)

    q = q / np.linalg.norm(q)                          # normalise the query
    M = M / np.linalg.norm(M, axis=1, keepdims=True)   # normalise each chunk

    scores = M @ q                                     # one dot product per chunk
    order = np.argsort(-scores)                        # descending = best first
    return [(int(i), float(scores[i])) for i in order]


def reciprocal_rank(ranked_ids, relevant_ids):
    """1 / (position of the first correct chunk); 0 if it never appears.

    Rewards ranking the right answer HIGH: #1 -> 1.0, #2 -> 0.5, #4 -> 0.25.
    Averaged over all questions this is MRR (Mean Reciprocal Rank).
    """
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / position
    return 0.0


def hit_at_k(ranked_ids, relevant_ids, k):
    """1 if a correct chunk is anywhere in the top k, else 0 (did we find it?)."""
    return 1.0 if any(cid in relevant_ids for cid in ranked_ids[:k]) else 0.0


def evaluate():
    # ---- Step 1: precompute everything that doesn't depend on the query. -----
    corpus_texts = [c["text"] for c in CORPUS]
    corpus_vectors = generate_embeddings_batch(corpus_texts)          # dense side
    bm25 = BM25([tokenize(t) for t in corpus_texts])                 # sparse side
    text_by_id = {c["id"]: c["text"] for c in CORPUS}

    metrics = {
        name: {"hit@1": [], "hit@3": [], "hit@5": [], "mrr": []} for name in CONFIGS
    }

    def make_points(ids, score_by_id):
        # Wrap ids as tiny stand-ins for Qdrant ScoredPoints (.score + .payload)
        # so we can hand them to the REAL rerank() unchanged.
        return [
            SimpleNamespace(
                score=score_by_id.get(cid, 0.0),
                payload={"text": text_by_id[cid], "chunk_id": cid},
            )
            for cid in ids
        ]

    for q in QUESTIONS:
        relevant = set(q["relevant_ids"])

        # ---- Dense ranking over the whole corpus. ------------------------
        ranked = cosine_rank(generate_embeddings(q["question"]), corpus_vectors)
        dense_score_by_id = {CORPUS[idx]["id"]: score for idx, score in ranked}
        dense_pool_ids = [CORPUS[idx]["id"] for idx, _ in ranked[:CANDIDATE_POOL]]

        # ---- Sparse (BM25) ranking over the same corpus. -----------------
        sparse_scores = bm25.scores(q["question"])
        sparse_order = sorted(
            range(len(CORPUS)), key=lambda i: sparse_scores[i], reverse=True
        )
        sparse_pool_ids = [CORPUS[i]["id"] for i in sparse_order[:CANDIDATE_POOL]]

        # ---- Fuse the two recall lists with RRF. -------------------------
        fused_pool_ids = reciprocal_rank_fusion([dense_pool_ids, sparse_pool_ids])

        # ---- Produce a top-k id list for each of the four configs. -------
        results = {
            "vector only":   dense_pool_ids[:FINAL_TOP_K],
            "hybrid only":   fused_pool_ids[:FINAL_TOP_K],
            "vector+rerank": [
                p.payload["chunk_id"]
                for p in rerank(
                    q["question"],
                    make_points(dense_pool_ids, dense_score_by_id),
                    FINAL_TOP_K,
                )
            ],
            "hybrid+rerank": [
                p.payload["chunk_id"]
                for p in rerank(
                    q["question"],
                    make_points(fused_pool_ids[:CANDIDATE_POOL], dense_score_by_id),
                    FINAL_TOP_K,
                )
            ],
        }

        # ---- Grade every config against the gold answer. ----------------
        for name, ids in results.items():
            metrics[name]["hit@1"].append(hit_at_k(ids, relevant, 1))
            metrics[name]["hit@3"].append(hit_at_k(ids, relevant, 3))
            metrics[name]["hit@5"].append(hit_at_k(ids, relevant, 5))
            metrics[name]["mrr"].append(reciprocal_rank(ids, relevant))

        print(f"Q: {q['question']}")
        print(f"   gold={list(relevant)}  vec={results['vector only']}  "
              f"hyb={results['hybrid only']}  hyb+rr={results['hybrid+rerank']}")

    # ---- Step 4: average each metric and print the comparison table. -----
    header = f"{'metric':<8}" + "".join(f"{name:>16}" for name in CONFIGS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for metric in ("hit@1", "hit@3", "hit@5", "mrr"):
        row = f"{metric:<8}"
        for name in CONFIGS:
            row += f"{sum(metrics[name][metric]) / len(QUESTIONS):>16.3f}"
        print(row)
    print("=" * len(header))
    print(f"\n{len(QUESTIONS)} questions over {len(CORPUS)} chunks.")


if __name__ == "__main__":
    evaluate()
