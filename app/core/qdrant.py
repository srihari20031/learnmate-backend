from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from app.core.config import settings

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)
# the size is based on the model used for embeddings, in this case "all-MiniLM-L6-v2" which has 384 dimensions
# cosine is for finding the angle between vectors, which is useful for semantic similarity, for rag always choose cosine
COLLECTION_NAME = "learnmate"


def ensure_collection():
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    for field_name in ("user_email", "session_id"):
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema="keyword",
                wait=True,
            )
        except Exception:
            pass


def create_collection_if_not_exists():
    ensure_collection()