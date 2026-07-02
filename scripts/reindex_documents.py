"""
One-off migration: re-index documents that were chunked before the 256-token fix.

WHY
---
The original pipeline chunked at 1000 tokens, but the embedding model truncates
at 256 — so old chunks were embedded using only their first ~25%. The fix to
chunk_text() only affects documents indexed *after* it. Documents indexed before
still sit in Qdrant with crippled embeddings and retrieve badly forever.

The stored chunk *text* is fine (full 1000-token chunks); only the *embedding*
was truncated. So we can recover the source text by stitching the old chunks
back together (detecting their overlap dynamically, so we don't have to assume
the old chunk_size/overlap), re-chunk at the current 256-token size, and
re-embed. No re-upload needed.

USAGE
-----
    python -m scripts.reindex_documents            # DRY RUN — reports only
    python -m scripts.reindex_documents --apply    # actually re-index
    python -m scripts.reindex_documents --apply --force   # include already-small docs

A document is skipped when its largest chunk is already <= 256 tokens (i.e. it
was indexed after the fix), unless --force is given.
"""

import argparse
import asyncio

from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

from app.core.qdrant import async_qdrant_client, COLLECTION_NAME
from app.database import documents_collection
from app.services.chunk_service import tokenizer, chunk_text
from app.services.rag_service import index_document

TOKEN_LIMIT = 256  # the embedding model's max sequence length


def _token_len(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def reconstruct_text(chunk_texts: list[str]) -> str:
    """Stitch overlapping chunks back into the original token stream.

    For each consecutive pair we find the largest suffix/prefix token overlap and
    drop the duplicate. This recovers the source regardless of the old overlap
    setting, so re-chunking produces clean, non-duplicated 256-token chunks.
    """
    if not chunk_texts:
        return ""

    token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in chunk_texts]
    merged = list(token_lists[0])

    for toks in token_lists[1:]:
        max_overlap = min(len(merged), len(toks))
        overlap = 0
        for cand in range(max_overlap, 0, -1):
            if merged[-cand:] == toks[:cand]:
                overlap = cand
                break
        merged.extend(toks[overlap:])

    return tokenizer.decode(merged)


async def _fetch_document_chunks(document_id: str) -> list[dict]:
    cursor = documents_collection.find({"document_id": document_id}).sort("chunk_index", 1)
    return [c async for c in cursor]


async def _delete_old(document_id: str) -> None:
    # Remove old vectors (by payload filter) and old Mongo chunk rows so the
    # re-index doesn't leave stale duplicates behind.
    await async_qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            )
        ),
    )
    await documents_collection.delete_many({"document_id": document_id})


async def migrate(apply: bool, force: bool) -> None:
    document_ids = await documents_collection.distinct("document_id")
    print(f"Found {len(document_ids)} document(s).\n")

    reindexed = skipped = failed = 0

    for document_id in document_ids:
        chunks = await _fetch_document_chunks(document_id)
        if not chunks:
            continue

        max_tokens = max(_token_len(c.get("text", "")) for c in chunks)

        if max_tokens <= TOKEN_LIMIT and not force:
            print(f"SKIP  {document_id}  ({len(chunks)} chunks, max {max_tokens} tok — already fine)")
            skipped += 1
            continue

        user_email = chunks[0].get("user_email")
        session_id = chunks[0].get("session_id")
        text = reconstruct_text([c.get("text", "") for c in chunks])
        new_chunks = chunk_text(text)

        print(
            f"REIDX {document_id}  {len(chunks)} chunks (max {max_tokens} tok) "
            f"-> {len(new_chunks)} chunks of <= {TOKEN_LIMIT} tok"
        )

        if not apply:
            continue

        try:
            await _delete_old(document_id)
            await index_document(
                document_id=document_id,
                user_email=user_email,
                text=text,
                session_id=session_id,
            )
            reindexed += 1
        except Exception as e:
            print(f"      FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    mode = "APPLIED" if apply else "DRY RUN (no changes written)"
    print(f"{mode}")
    print(f"  re-indexed: {reindexed}")
    print(f"  skipped:    {skipped}")
    print(f"  failed:     {failed}")
    print("=" * 50)
    if not apply:
        print("Re-run with --apply to perform the migration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index pre-256-token documents.")
    parser.add_argument("--apply", action="store_true", help="actually write changes")
    parser.add_argument("--force", action="store_true", help="include already-small docs")
    args = parser.parse_args()

    asyncio.run(migrate(apply=args.apply, force=args.force))
