from upstash_redis.asyncio import Redis
from app.core.config import settings


# Async client: cache reads/writes are network I/O to Upstash, so awaiting them
# keeps the event loop free instead of blocking it on every get/set (matches the
# async Qdrant / Groq clients used elsewhere).
redis_client = Redis(
    settings.upstash_redis_url,
    token=settings.upstash_redis_token
)