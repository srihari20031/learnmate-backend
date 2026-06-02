from upstash_redis import Redis
from app.core.config import settings


redis_client = Redis(
    settings.upstash_redis_url,
    token=settings.upstash_redis_token  
)