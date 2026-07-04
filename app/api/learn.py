from fastapi import APIRouter, Header, Depends

from app.models.schema import ChatRequest, LearnResponse
from app.services.claude_service import get_claude_response, generate_curriculum_ai
from app.services.note_generator import generate_note
from app.services.notion_service import create_notion_topic, create_user_parent_page
from app.core.session import get_context, get_chat, update_last_message
from app.core.security import get_current_user
from app.models.user import TokenData
from app.database import users_collection
from app.core.config import settings
from app.services.cache_service import (
    get_cached_curriculum,
    set_cached_curriculum,
    get_cached_note,
    set_cached_note,
)

router = APIRouter()


async def generate_learning_notes(session_id: str, user_email: str) -> dict:
    await get_chat(session_id, user_email)

    context = await get_context(session_id, user_email)
    known_stack = context.get("known_stack") or "beginner"
    target_tech = context.get("target_tech")

    if not target_tech:
        return {
            "status": "chatting",
            "notion_urls": [],
            "notion_pages": [],
        }

    user_doc = await users_collection.find_one({"email": user_email})
    notion_token = user_doc.get("notion_access_token") if user_doc else None
    notion_parent_page_id = user_doc.get("notion_parent_page_id") if user_doc else None

    if notion_token and not notion_parent_page_id and known_stack and target_tech:
        try:
            notion_parent_page_id = create_user_parent_page(
                access_token=notion_token,
                target_tech=target_tech,
                known_stack=known_stack or "",
            )
            await users_collection.update_one(
                {"email": user_email},
                {"$set": {"notion_parent_page_id": notion_parent_page_id}},
            )
        except Exception as e:
            print(f"[Notion] Failed to create parent page for user: {e}")

    if not notion_token:
        notion_token = settings.notion_api_key
        notion_parent_page_id = settings.notion_parent_page_id

    topics = await get_cached_curriculum(known_stack, target_tech)
    if not topics:
        topics = await generate_curriculum_ai(known_stack, target_tech)
        await set_cached_curriculum(known_stack, target_tech, topics)

    urls = []
    notion_pages = []
    for topic in topics:
        note_content = await get_cached_note(topic, known_stack, target_tech)
        if not note_content:
            note_content = await generate_note(topic, known_stack, target_tech)
            await set_cached_note(topic, known_stack, target_tech, note_content)

        url = await create_notion_topic(
            title=topic,
            content=note_content,
            session_id=session_id,
            notion_token=notion_token,
            page_id=notion_parent_page_id,
        )
        urls.append(url)
        notion_pages.append({"title": topic, "url": url})

    # Persist the links onto the assistant message that triggered generation, so
    # they survive a reload (they aren't known when that message is first saved).
    await update_last_message(
        session_id,
        user_email,
        {"notion_urls": urls, "notion_pages": notion_pages},
    )

    return {
        "status": "completed",
        "notion_urls": urls,
        "notion_pages": notion_pages,
    }


@router.post("/message")
async def learn(
    request: ChatRequest,
    session_id: str = Header(...),
    current_user: TokenData = Depends(get_current_user)
):
    response, sources = await get_claude_response(session_id, request.message, current_user.email)

    if "READY_TO_GENERATE" not in response.upper():
        return LearnResponse(
            response=response,
            session_id=session_id,
            status="chatting",
            notion_urls=[],
            sources=sources,
        )

    result = await generate_learning_notes(session_id, current_user.email)

    return LearnResponse(
        response="Your notes are ready in Notion!",
        session_id=session_id,
        status=result["status"],
        notion_urls=result["notion_urls"],
        notion_pages=result.get("notion_pages", []),
    )
