from app.database import sessions_collection, contexts_collection
from app.core.security import get_current_user
from fastapi import Header, HTTPException

async def get_session(session_id: str, current_user_email: str) -> list:
    # Verify session belongs to current user
    doc = await sessions_collection.find_one({"_id": session_id, "user_email": current_user_email})
    if not doc:
        # Create new session for this user
        await sessions_collection.insert_one({
            "_id": session_id, 
            "user_email": current_user_email,
            "messages": []
        })
        return []
    return doc.get("messages", [])

async def add_message(session_id: str, role: str, content: str, current_user_email: str):
    await sessions_collection.update_one(
        {"_id": session_id, "user_email": current_user_email},
        {"$push": {"messages": {"role": role, "content": content}}},
        upsert=True
    )

async def clear_session(session_id: str, current_user_email: str):
    await sessions_collection.delete_one({"_id": session_id, "user_email": current_user_email})
    await contexts_collection.delete_one({"_id": session_id, "user_email": current_user_email})

async def set_context(session_id: str, key: str, value: str, current_user_email: str):
    await contexts_collection.update_one(
        {"_id": session_id, "user_email": current_user_email},
        {"$set": {f"data.{key}": value}},
        upsert=True
    )

async def get_context(session_id: str, current_user_email: str) -> dict:
    doc = await contexts_collection.find_one({"_id": session_id, "user_email": current_user_email})
    return doc.get("data", {}) if doc else {}