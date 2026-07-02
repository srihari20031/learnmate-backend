from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException
from app.database import chats_collection, contexts_collection


CHAT_NOT_FOUND = "Chat not found. Create one first via POST /chats."


def serialize_chat(doc: dict) -> dict:
    return {
        "chat_id": doc["_id"],
        "session_id": doc["_id"],
        "title": doc.get("title") or "New Chat",
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


async def create_chat(title: str, current_user_email: str) -> dict:
    now = datetime.now(timezone.utc)
    chat_id = str(uuid4())
    doc = {
        "_id": chat_id,
        "user_email": current_user_email,
        "title": title or "New Chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    await chats_collection.insert_one(doc)
    return serialize_chat(doc)


async def list_chats(current_user_email: str, limit: int = 50) -> list:
    cursor = chats_collection.find(
        {"user_email": current_user_email}
    ).sort("updated_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [serialize_chat(doc) for doc in docs]


async def get_chat(chat_id: str, current_user_email: str) -> dict:
    doc = await chats_collection.find_one({"_id": chat_id, "user_email": current_user_email})
    if not doc:
        raise HTTPException(status_code=404, detail=CHAT_NOT_FOUND)
    return serialize_chat(doc)


async def update_chat_title(chat_id: str, title: str, current_user_email: str) -> dict:
    now = datetime.now(timezone.utc)
    result = await chats_collection.update_one(
        {"_id": chat_id, "user_email": current_user_email},
        {
            "$set": {
                "title": title.strip() or "New Chat",
                "updated_at": now,
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=CHAT_NOT_FOUND)

    doc = await chats_collection.find_one({"_id": chat_id, "user_email": current_user_email})
    if not doc:
        raise HTTPException(status_code=404, detail=CHAT_NOT_FOUND)
    return serialize_chat(doc)


async def get_session(session_id: str, current_user_email: str) -> list:
    doc = await chats_collection.find_one({"_id": session_id, "user_email": current_user_email})
    if not doc:
        raise HTTPException(status_code=404, detail=CHAT_NOT_FOUND)
    return [
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "sent_at": message.get("sent_at"),
            "attachments": message.get("attachments", []),
        }
        for message in doc.get("messages", [])
    ]


async def add_message(session_id: str, role: str, content: str, current_user_email: str, attachments: list[dict] | None = None):
    now = datetime.now(timezone.utc)
    message_doc = {
        "role": role,
        "content": content,
        "sent_at": now,
    }
    if attachments is not None:
        message_doc["attachments"] = attachments
    result = await chats_collection.update_one(
        {"_id": session_id, "user_email": current_user_email},
        {
            "$push": {"messages": message_doc},
            "$set": {"updated_at": now},
        },
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=CHAT_NOT_FOUND)


async def clear_session(session_id: str, current_user_email: str):
    # Local import: keeps this module free of the embedding-model import chain
    # that app.services.rag_service pulls in at import time.
    from app.services.rag_service import delete_session_documents

    # Remove uploaded documents (Mongo chunks + metadata + Qdrant vectors) so a
    # reset doesn't leave orphaned data behind.
    await delete_session_documents(current_user_email, session_id)

    await chats_collection.delete_one({"_id": session_id, "user_email": current_user_email})
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
