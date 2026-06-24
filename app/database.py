from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = AsyncIOMotorClient(settings.mongodb_uri)
db = client["learnnewcluster"]

users_collection = db["users"]
chats_collection = db["chats"]
generated_notes_collection = db["generated_notes"]
contexts_collection = db["session_contexts"]
documents_collection = db["documents"]
documents_metadata_collection = db["documents_metadata"]

async def close_db():
    client.close()