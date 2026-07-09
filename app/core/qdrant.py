from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams, Distance
from app.core.config import settings

# Async client: Qdrant calls are network I/O, so the async client lets them run
# on the event loop without consuming a worker thread (unlike the sync client +
# asyncio.to_thread). CPU-bound work like embedding still uses to_thread.
async_qdrant_client = AsyncQdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

# the size is based on the model used for embeddings, in this case "all-MiniLM-L6-v2" which has 384 dimensions
# cosine is for finding the angle between vectors, which is useful for semantic similarity, for rag always choose cosine
COLLECTION_NAME = "learnmate"


async def ensure_collection():
    # Called once at app startup, not per request, so the collection and its
    # payload indexes are set up a single time instead of on every embed/search.
    if not await async_qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        await async_qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # document_id is indexed too so a SINGLE document's vectors can be deleted by
    # payload filter (Qdrant rejects a filter on an unindexed keyword field).
    for field_name in ("user_email", "session_id", "document_id"):
        try:
            await async_qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema="keyword",
                wait=True,
            )
        except Exception:
            pass
