from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.services.notion_service import create_notion_topic
from app.core.security import get_current_user
from app.models.user import TokenData
from app.database import users_collection
from app.core.config import settings
import secrets
from urllib.parse import urlencode

router = APIRouter()

NOTION_AUTH_URL = "https://api.notion.com/v1/oauth/authorize"

from app.api.notion_oauth_state import oauth_states


@router.get("/status")
async def notion_status(current_user: TokenData = Depends(get_current_user)):
    user_doc = await users_collection.find_one({"email": current_user.email})
    connected = bool(user_doc and user_doc.get("notion_access_token"))
    return {"connected": connected}


@router.get("/connect")
async def notion_connect(current_user: TokenData = Depends(get_current_user)):
    redirect_uri = f"{settings.FRONTEND_URL.rstrip('/')}/notion/callback"
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"email": current_user.email, "redirect_uri": redirect_uri}
    params = {
        "client_id": settings.NOTION_CLIENT_ID,
        "response_type": "code",
        "owner": "user",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    authorization_url = f"{NOTION_AUTH_URL}?{urlencode(params)}"
    return {"authorization_url": authorization_url}


@router.post("/disconnect")
async def notion_disconnect(current_user: TokenData = Depends(get_current_user)):
    await users_collection.update_one(
        {"email": current_user.email},
        {"$unset": {
            "notion_access_token": "",
            "notion_refresh_token": "",
            "notion_workspace_id": "",
        }},
    )
    return {"status": "success", "message": "Notion disconnected"}


@router.post("/create-topic")
async def create_topic(
    title: str,
    content: str,
    session_id: str = None,
    current_user: TokenData = Depends(get_current_user),
):
    user_email = current_user.email
    notion_token = None
    if user_email:
        user_doc = await users_collection.find_one({"email": user_email})
        if user_doc:
            notion_token = user_doc.get("notion_access_token")

    notion_url = create_notion_topic(title, content, session_id, notion_token)
    return {"status": "ok", "notion_url": notion_url}
