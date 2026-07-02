# app/services/rag_service.py

import asyncio
from datetime import datetime
from types import SimpleNamespace
from bson import ObjectId
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

from app.database import (
    documents_collection,
    documents_metadata_collection,
)
from app.core.qdrant import async_qdrant_client, COLLECTION_NAME
from app.services.chunk_service import chunk_text
from app.services.embedding import (
        generate_embeddings_batch,
        store_embeddings_batch,
    )
from app.services.keyword_service import invalidate_bm25
from app.models.schema import DocumentStatus


async def save_document_metadata(
    user_email: str,
    filename: str,
    file_type: str,
):
    document = {
        "user_email": user_email,
        "filename": filename,
        "file_type": file_type,
        "status": DocumentStatus.processing.value,
        "chunk_count": 0,
        "uploaded_at": datetime.utcnow(),
    }

    result = await documents_metadata_collection.insert_one(document)

    document_id = str(result.inserted_id)

    document["_id"] = document_id
    document["id"] = document_id

    return SimpleNamespace(**document)


async def delete_session_documents(user_email: str, session_id: str | None) -> dict:
    # Remove a session's uploaded documents everywhere they live: chunk rows and
    # metadata in Mongo, and vectors in Qdrant. Without this, resetting a session
    # orphans the chunks + embeddings forever (and they'd resurface if the
    # session id were ever reused).
    if not session_id:
        return {"documents": 0, "chunks": 0}

    scope = {"user_email": user_email, "session_id": session_id}

    # Metadata has no session_id field, so gather the doc ids from the chunks
    # first (before we delete them) to clean up metadata by _id afterwards.
    document_ids = await documents_collection.distinct("document_id", scope)

    # Vectors: delete by payload filter (both fields are indexed in Qdrant).
    await async_qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="user_email", match=MatchValue(value=user_email)),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                ]
            )
        ),
    )

    chunk_result = await documents_collection.delete_many(scope)

    if document_ids:
        await documents_metadata_collection.delete_many(
            {"_id": {"$in": [ObjectId(d) for d in document_ids]}}
        )

    # Drop the now-stale BM25 index for this session (cache is otherwise
    # self-correcting via its signature, but no reason to hold the memory).
    invalidate_bm25(session_id)

    return {"documents": len(document_ids), "chunks": chunk_result.deleted_count}


async def get_session_chunks(user_email: str, session_id: str | None) -> list[dict]:
    # All chunks for a session, as {chunk_id, text}. Hybrid search needs the full
    # corpus (not just vector hits) so BM25 can surface chunks vector search
    # missed. chunk_id is rebuilt to match the id stored in Qdrant's payload.
    if not session_id:
        return []

    # Stable sort order so the BM25 cache signature (the ordered chunk ids) is
    # consistent across reads — otherwise a reordered read looks like a new
    # corpus and forces a needless rebuild.
    cursor = documents_collection.find(
        {"user_email": user_email, "session_id": session_id},
        projection={"_id": 0, "document_id": 1, "chunk_index": 1, "text": 1},
    ).sort([("document_id", 1), ("chunk_index", 1)])

    chunks = []
    async for doc in cursor:
        chunks.append(
            {
                "chunk_id": f"{doc['document_id']}_{doc['chunk_index']}",
                "text": doc.get("text", ""),
            }
        )
    return chunks


async def index_document(
    document_id: str,
    user_email: str,
    text: str,
    session_id: str | None = None,
):
    try:
        chunks = chunk_text(text)

        if not chunks:
            await documents_metadata_collection.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$set": {
                        "status": DocumentStatus.ready.value,
                        "chunk_count": 0,
                    }
                },
            )
            return {
                "status": "ready",
                "document_id": document_id,
                "chunks_indexed": 0,
            }

        # Embed every chunk in one batched call, offloaded to a worker thread so
        # the model's CPU/GPU work does not block the asyncio event loop.
        embeddings = await asyncio.to_thread(generate_embeddings_batch, chunks)

        now = datetime.utcnow()

        # One bulk insert instead of a write per chunk.
        await documents_collection.insert_many(
            [
                {
                    "document_id": document_id,
                    "user_email": user_email,
                    "session_id": session_id,
                    "chunk_index": i,
                    "text": chunk,
                    "created_at": now,
                }
                for i, chunk in enumerate(chunks)
            ]
        )

        points = [
            {
                "chunk_id": f"{document_id}_{i}",
                "embedding": embeddings[i],
                "payload": {
                    "document_id": document_id,
                    "user_email": user_email,
                    "session_id": session_id,
                    "chunk_index": i,
                    "text": chunk,
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        # Single batched Qdrant upsert via the async client (network I/O, so no
        # worker thread needed — unlike the CPU-bound embedding above).
        await store_embeddings_batch(points)

        await documents_metadata_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": DocumentStatus.ready.value,
                    "chunk_count": len(chunks),
                }
            },
        )

        return {
            "status": "ready",
            "document_id": document_id,
            "chunks_indexed": len(chunks),
        }

    except Exception as e:

        await documents_metadata_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": DocumentStatus.failed.value,
                    "error": str(e),
                }
            },
        )

        raise