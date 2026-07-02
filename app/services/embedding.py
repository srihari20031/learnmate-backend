from sentence_transformers import SentenceTransformer
from app.core.qdrant import async_qdrant_client, COLLECTION_NAME
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(text: str) -> list[float]:
    return model.encode(text).tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    # Encoding a list in one call lets the model batch the work on the GPU/CPU,
    # which is far faster than calling encode() once per chunk in a Python loop.
    return model.encode(texts).tolist()


async def store_embedding(chunk_id: str, embedding: list[float], payload: dict):
    try:
        return await async_qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=embedding,
                    payload={**payload, "chunk_id": chunk_id},
                )
            ],
        )
    except Exception as e:
        raise Exception(f"Failed to store embedding: {str(e)}")


async def store_embeddings_batch(points: list[dict]):
    # Upsert every chunk's vector in a single Qdrant request instead of one
    # network round-trip per chunk. Each item is {chunk_id, embedding, payload}.
    try:
        return await async_qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, p["chunk_id"])),
                    vector=p["embedding"],
                    payload={**p["payload"], "chunk_id": p["chunk_id"]},
                )
                for p in points
            ],
        )
    except Exception as e:
        raise Exception(f"Failed to store embeddings: {str(e)}")


async def search_embeddings(
    query_vector: list[float],
    user_email: str,
    session_id: str,
    top_k: int = 5,
    min_score: float = 0.3
):
    response = await async_qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="user_email",
                    match=MatchValue(value=user_email),
                ),
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                ),
            ]
        ),
        limit=top_k,
        with_payload=True,
        score_threshold=min_score
    )
    return response.points
