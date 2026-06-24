# app/services/rag_service.py

from datetime import datetime
from types import SimpleNamespace
from bson import ObjectId

from app.database import (
    documents_collection,
    documents_metadata_collection,
)
from app.services.chunk_service import chunk_text
from app.services.embedding import (
        generate_embeddings,
        store_embedding,
    )
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


async def index_document(
    document_id: str,
    user_email: str,
    text: str,
    session_id: str | None = None,
):
    try:
        chunks = chunk_text(text)

        success_count = 0

        for i, chunk in enumerate(chunks):

            chunk_id = f"{document_id}_{i}"

            await documents_collection.insert_one(
                {
                    "document_id": document_id,
                    "user_email": user_email,
                    "session_id": session_id,
                    "chunk_index": i,
                    "text": chunk,
                    "created_at": datetime.utcnow(),
                }
            )

            embedding = generate_embeddings(chunk)

            store_result = store_embedding(
                chunk_id=chunk_id,
                embedding=embedding,
                payload={
                    "document_id": document_id,
                    "user_email": user_email,
                    "session_id": session_id,
                    "chunk_index": i,
                    "text": chunk
                },
            )

            if store_result:
                success_count += 1

        if success_count == len(chunks):

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

        await documents_metadata_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": DocumentStatus.failed.value,
                }
            },
        )

        return {
            "status": "failed",
            "document_id": document_id,
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