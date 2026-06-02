from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = AsyncIOMotorClient(settings.mongodb_uri)
db = client["learnnewcluster"]

users_collection = db["users"]
sessions_collection = db["sessions"]
messages_collection = db["messages"]
generated_notes_collection = db["generated_notes"]
contexts_collection = db["session_contexts"]

async def close_db():
    client.close()