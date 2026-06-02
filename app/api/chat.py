# app/api/chat.py

from fastapi import APIRouter, Header, Depends, HTTPException
from app.core.session import get_session, clear_session, add_message
from app.models.schema import ChatRequest, ChatResponse
from app.services.claude_service import get_claude_response
from app.core.security import get_current_user
from app.models.user import TokenData

router = APIRouter()

async def verify_session_ownership(session_id: str, current_user: TokenData):
    # In a full implementation, we would check if the session belongs to the user
    # For now, we'll rely on the session functions to enforce ownership
    pass

@router.post("/message")
async def message(request: ChatRequest, session_id: str = Header(...), current_user: TokenData = Depends(get_current_user)) -> ChatResponse:
    claude_response = await get_claude_response(session_id=session_id, user_message=request.message, current_user_email=current_user.email)
    return ChatResponse(response=claude_response, session_id=session_id)


@router.get("/session/{session_id}")
async def get_session_messages(session_id: str, current_user: TokenData = Depends(get_current_user)):
    session_messages = await get_session(session_id, current_user.email)
    return {"session_id": session_id, "messages": session_messages}

@router.post("/reset/{session_id}")
async def reset_session(session_id: str, current_user: TokenData = Depends(get_current_user)):
    await clear_session(session_id, current_user.email)
    return {"status": "ok", "message": f"Session {session_id} cleared"}