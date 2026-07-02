# app/api/chat.py

from uuid import uuid4
from fastapi import APIRouter, Header, Depends, Body, UploadFile, File, Form
from app.core.session import get_session, clear_session, create_chat, list_chats, get_chat, update_chat_title
from app.models.schema import ChatResponse, ChatCreateRequest, ChatCreateResponse, ChatListResponse, ChatUpdateTitleRequest, SessionMessagesResponse
from app.core.session import set_context, get_context
from app.services.claude_service import get_claude_response
from app.api.learn import generate_learning_notes
from app.services.file_service import process_uploaded_file
from app.core.security import get_current_user
from app.models.user import TokenData
from app.services.rag_service import index_document, save_document_metadata

router = APIRouter()


@router.post("/chats", response_model=ChatCreateResponse)
async def create_chat_session(
    request: ChatCreateRequest = Body(default=None),
    current_user: TokenData = Depends(get_current_user),
):
    title = request.title if request else "New Chat"
    return await create_chat(title, current_user.email)


@router.get("/chats", response_model=ChatListResponse)
async def get_chat_sessions(current_user: TokenData = Depends(get_current_user)):
    chats = await list_chats(current_user.email)
    return {"chats": chats}


@router.get("/chats/{chat_id}", response_model=ChatCreateResponse)
async def get_chat_session(chat_id: str, current_user: TokenData = Depends(get_current_user)):
    return await get_chat(chat_id, current_user.email)


@router.patch("/chats/{chat_id}/title", response_model=ChatCreateResponse)
async def update_chat_session_title(
    chat_id: str,
    request: ChatUpdateTitleRequest = Body(...),
    current_user: TokenData = Depends(get_current_user),
):
    return await update_chat_title(chat_id, request.title, current_user.email)


@router.post("/message")
async def message(
    message: str = Form(...),
    session_id: str = Header(...),
    current_user: TokenData = Depends(get_current_user),
    attachments: list[UploadFile] = File(default=[]),
) -> ChatResponse:
    await get_chat(session_id, current_user.email)

    processed_attachments = []
    
    for file in attachments:
        result = await process_uploaded_file(file)
        
        att_id = str(uuid4())
        
        if result.get("type") == "document":
            document = await save_document_metadata(
                user_email=current_user.email,
                filename=result.get("filename"),
                file_type=result.get("type")
            )
            
            await index_document(
                document_id=document.id,
                user_email=current_user.email,
                text=result.get("text", ""),
                session_id=session_id
            )
            
            processed_attachments.append({
                "id": att_id,
                "filename": result.get("filename"),
                "type": "document",
                "mime_type": result.get("mime", "application/octet-stream"),
                "text": result.get("text"),
            })
        
        elif result.get("type") == "image":
            processed_attachments.append({
                "id": att_id,
                "filename": result.get("filename"),
                "type": "image",
                "mime_type": result.get("mime_type"),
                "base64": result.get("base64"),
            })
    
    # We deliberately do NOT staple the extracted file text onto the user
    # message. The document is already indexed for retrieval above, so the
    # model receives its content through RAG (see get_claude_response). Inlining
    # the raw text here would also leak the entire file dump into the visible
    # user bubble when the frontend reloads chat history.
    claude_response = await get_claude_response(
        session_id=session_id,
        user_message=message,
        current_user_email=current_user.email,
        attachments=processed_attachments if processed_attachments else None,
    )

    if "READY_TO_GENERATE" in claude_response.upper():
        result = await generate_learning_notes(session_id, current_user.email)
        return ChatResponse(
            response="Your notes are ready in Notion!",
            session_id=session_id,
            status=result["status"],
            notion_urls=result["notion_urls"],
        )

    return ChatResponse(response=claude_response, session_id=session_id)


@router.post("/upload")
async def upload_file(session_id: str = Header(...), file: UploadFile = File(...), current_user: TokenData = Depends(get_current_user)):
    await get_chat(session_id, current_user.email)
    result = await process_uploaded_file(file)
    if result.get("type") == "document":
        document = await save_document_metadata(
            user_email=current_user.email,
            filename=result.get("filename"),
            file_type=result.get("type")
        )
        
        await index_document(
            document_id=document.id,
            user_email=current_user.email,
            text=result.get("text", ""),
            session_id=session_id
        )
        
    return {"filename": result.get("filename"), "type": result.get("type"), "status": "uploaded"}


@router.get("/session/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: TokenData = Depends(get_current_user)):
    session_messages = await get_session(session_id, current_user.email)
    return {"session_id": session_id, "messages": session_messages}

@router.post("/reset/{session_id}")
async def reset_session(session_id: str, current_user: TokenData = Depends(get_current_user)):
    await clear_session(session_id, current_user.email)
    return {"status": "ok", "message": f"Session {session_id} cleared"}
