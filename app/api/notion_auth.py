from fastapi import APIRouter, Depends, HTTPException, Body
from app.core.security import get_current_user, TokenData
from app.database import users_collection
from app.core.config import settings
import httpx
from app.api.notion_oauth_state import oauth_states

router = APIRouter()

NOTION_TOKEN_URL = "https://api.notion.com/v1/oauth/token"


@router.post("/callback")
async def notion_callback(
    code: str = Body(..., embed=True),
    state: str = Body(..., embed=True)
):
    stored = oauth_states.pop(state, None)
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid or expired state")

    user_email = stored["email"]
    redirect_uri = stored["redirect_uri"]

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }

    auth = (settings.NOTION_CLIENT_ID, settings.NOTION_CLIENT_SECRET)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            NOTION_TOKEN_URL,
            data=token_data,
            auth=auth
        )

        if response.status_code != 200:
            print(f"[Notion] Token exchange failed: {response.status_code} {response.text}")
            raise HTTPException(
                status_code=400,
                detail="Failed to exchange code for token"
            )

        token_response = response.json()

    access_token = token_response.get("access_token")
    workspace_id = token_response.get("workspace_id")

    await users_collection.update_one(
        {"email": user_email},
        {"$set": {
            "notion_access_token": access_token,
            "notion_workspace_id": workspace_id,
        }},
    )

    return {
        "status": "success",
        "message": "Notion connected successfully",
        "workspace_id": workspace_id,
    }


@router.post("/notion/disconnect")
async def notion_disconnect(current_user: TokenData = Depends(get_current_user)):
    await users_collection.update_one(
        {"email": current_user.email},
        {"$unset": {
            "notion_access_token": "",
            "notion_refresh_token": "",
            "notion_workspace_id": "",
            "notion_parent_page_id": "",
        }},
    )
    return {"status": "success", "message": "Notion disconnected"}
