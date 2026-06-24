from sentence_transformers import SentenceTransformer
from app.core.qdrant import qdrant_client, ensure_collection
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

def store_embedding(chunk_id: str, embedding: list[float], payload: dict):
    ensure_collection()
    try:
        result = qdrant_client.upsert(
            collection_name="learnmate",
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=embedding,
                    payload={**payload, "chunk_id": chunk_id},
                )
            ],
        )
        return result

    except Exception as e:
        raise Exception(f"Failed to store embedding: {str(e)}")
    
def search_embeddings(
    query_vector: list[float],
    user_email: str,
    session_id: str,
    top_k: int = 5,
    min_score: float = 0.3
):
    ensure_collection()
    response = qdrant_client.query_points(
        collection_name="learnmate",
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
    for r in response.points:
        print("Checking", r.score, r.payload.get("text")[:50])
    return response.points