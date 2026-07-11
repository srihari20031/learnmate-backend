# app/services/profile_service.py
#
# The PROFILE lane — deliberately separate from RAG.
#
# A resume answers "who is this user?" (asked once, true for every chat), not
# "what does this document say about X?" (asked per query). So we do NOT chunk,
# embed, or store it in Qdrant — that only pollutes retrieval and wastes storage.
# Instead we extract the user's known tech stack ONCE and store it on the user,
# so every one of their chats can use it without re-uploading.

import json
import logging
from datetime import datetime

from app.core.session import get_context
from app.database import profiles_collection
from app.services.llm import ai_invoke

# A resume/CV is small; cap the text we feed the extractor so token cost stays
# bounded even for an oversized upload.
MAX_PROFILE_CHARS = 12_000

logger = logging.getLogger(__name__)


def merge_stacks(existing: str | None, incoming: str | None) -> str:
    # Union two comma-separated tech lists into one clean list: existing items
    # first (order preserved), then genuinely-new items from incoming. Dedup is
    # case-insensitive and applies both across and WITHIN the inputs, and each
    # item is whitespace-normalized. Lets the resume-derived stack and the
    # conversation-derived stack accumulate instead of overwriting each other.
    seen: set[str] = set()
    result: list[str] = []
    for group in ((existing or ""), (incoming or "")):
        for item in group.split(","):
            item = item.strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return ", ".join(result)


async def extract_stack(text: str) -> dict:
    # Classify + extract in one call. Returns {"is_profile": bool,
    # "known_stack": str | None}. The classification lets the profile endpoint
    # reject a non-resume upload (e.g. someone drops study notes into the resume
    # slot) instead of inventing a stack from it.
    if not text or not text.strip():
        return {"is_profile": False, "known_stack": None}

    prompt = f"""You are analyzing an uploaded document to decide whether it describes a PERSON'S own background (a resume, CV, or LinkedIn-style profile) or whether it is something else (study material, an article, notes about a topic).

Document:
\"\"\"
{text[:MAX_PROFILE_CHARS]}
\"\"\"

Return ONLY JSON, nothing else:
- If it is a resume / CV / personal profile:
  {{"is_profile": true, "known_stack": "comma-separated technologies, languages, frameworks, databases and tools the person has hands-on experience with"}}
- Otherwise:
  {{"is_profile": false, "known_stack": null}}

Only include technologies the person clearly has real experience with (skills, projects, work history). Do not include tools merely mentioned in passing."""

    try:
        response = await ai_invoke(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = response
        data = json.loads(raw)
        
    except Exception:
        logger.exception("Stack extraction failed")
        return {"is_profile": False, "known_stack": None}

    return {
        "is_profile": bool(data.get("is_profile")),
        "known_stack": data.get("known_stack"),
    }


async def save_user_profile(
    user_email: str,
    known_stack: str,
    resume_filename: str | None = None,
) -> dict:
    # Upsert the user's profile. known_stack is MERGED with any prior stack so a
    # second resume upload adds to the picture instead of wiping it.
    existing = await profiles_collection.find_one({"_id": user_email})
    merged = merge_stacks(
        existing.get("known_stack") if existing else None, known_stack
    )

    doc = {
        "known_stack": merged,
        "resume_filename": resume_filename,
        "updated_at": datetime.utcnow(),
    }
    await profiles_collection.update_one(
        {"_id": user_email}, {"$set": doc}, upsert=True
    )
    doc["user_email"] = user_email
    return doc


async def get_user_profile(user_email: str) -> dict:
    doc = await profiles_collection.find_one({"_id": user_email})
    if not doc:
        return {}
    return {
        "user_email": user_email,
        "known_stack": doc.get("known_stack"),
        "resume_filename": doc.get("resume_filename"),
        "updated_at": doc.get("updated_at"),
    }


async def delete_user_profile(user_email: str) -> None:
    await profiles_collection.delete_one({"_id": user_email})


async def resolve_known_stack(session_id: str | None, user_email: str) -> str | None:
    # The effective known stack for a chat = the user-wide resume stack (base)
    # unioned with anything they've mentioned in THIS session's conversation.
    # Profile is the durable base; session context is the per-chat delta.
    profile = await get_user_profile(user_email)
    base = profile.get("known_stack")

    session_stack = None
    if session_id:
        ctx = await get_context(session_id, user_email)
        session_stack = ctx.get("known_stack")

    merged = merge_stacks(base, session_stack)
    return merged or None
