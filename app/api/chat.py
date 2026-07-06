# app/api/chat.py

import json
from uuid import uuid4
from fastapi import APIRouter, Header, Depends, Body, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from app.core.session import get_session, clear_session, create_chat, list_chats, get_chat, update_chat_title
from app.models.schema import ChatResponse, ChatCreateRequest, ChatCreateResponse, ChatListResponse, ChatUpdateTitleRequest, SessionMessagesResponse
from app.core.session import set_context, get_context
from app.services.claude_service import get_claude_response, get_claude_response_stream
from app.services.tutor_service import handle_turn
from app.services.file_service import process_uploaded_file
from app.core.security import get_current_user
from app.models.user import TokenData
from app.services.rag_service import index_document, save_document_metadata

router = APIRouter()


async def process_message_attachments(
    attachments: list[UploadFile], user_email: str, session_id: str
) -> list[dict]:
    # Extract, persist and index each attachment. Documents are chunked+embedded
    # for retrieval; images are passed through as base64. Shared by /message and
    # /message/stream.
    processed = []
    for file in attachments:
        result = await process_uploaded_file(file)
        att_id = str(uuid4())

        if result.get("type") == "document":
            document = await save_document_metadata(
                user_email=user_email,
                filename=result.get("filename"),
                file_type=result.get("type"),
            )
            await index_document(
                document_id=document.id,
                user_email=user_email,
                text=result.get("text", ""),
                session_id=session_id,
            )
            processed.append({
                "id": att_id,
                "filename": result.get("filename"),
                "type": "document",
                "mime_type": result.get("mime", "application/octet-stream"),
                "text": result.get("text"),
            })

        elif result.get("type") == "image":
            processed.append({
                "id": att_id,
                "filename": result.get("filename"),
                "type": "image",
                "mime_type": result.get("mime_type"),
                "base64": result.get("base64"),
            })

    return processed


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

    processed_attachments = await process_message_attachments(
        attachments, current_user.email, session_id
    )

    # We deliberately do NOT staple the extracted file text onto the user
    # message. The document is already indexed for retrieval, so the model
    # receives its content through RAG. The turn orchestrator handles both
    # conversational replies and the guided topic-by-topic teaching flow,
    # including the READY_TO_GENERATE / START_TEACHING / NEXT_TOPIC control
    # tokens and any Notion note generation.
    result = await handle_turn(
        session_id=session_id,
        user_message=message,
        user_email=current_user.email,
        attachments=processed_attachments if processed_attachments else None,
    )

    return ChatResponse(
        response=result["reply"],
        session_id=session_id,
        status=result["status"],
        sources=result.get("sources", []),
        notion_pages=result.get("notion_pages", []),
        notion_urls=[p["url"] for p in result.get("notion_pages", [])],
    )


@router.post("/message/stream")
async def message_stream(
    message: str = Form(...),
    session_id: str = Header(...),
    current_user: TokenData = Depends(get_current_user),
    attachments: list[UploadFile] = File(default=[]),
):
    # Server-Sent Events version of /message: streams the answer token-by-token.
    # Same contract (multipart form: `message` + optional `attachments`, plus the
    # `session-id` header). Response is text/event-stream instead of JSON.
    await get_chat(session_id, current_user.email)

    processed_attachments = await process_message_attachments(
        attachments, current_user.email, session_id
    )

    async def event_source():
        try:
            async for event in get_claude_response_stream(
                session_id=session_id,
                user_message=message,
                current_user_email=current_user.email,
                attachments=processed_attachments if processed_attachments else None,
            ):
                yield event
        except Exception as e:
            # Surface failures as a final SSE event instead of a dropped stream.
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy (nginx) response buffering
        },
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
    background_tasks: BackgroundTasks,
    session_id: str = Header(...),
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
):
    await get_chat(session_id, current_user.email)

    # Read/extract the file NOW — the UploadFile is tied to this request and may
    # be gone by the time the background task runs.
    result = await process_uploaded_file(file)

    if result.get("type") == "document":
        document = await save_document_metadata(
            user_email=current_user.email,
            filename=result.get("filename"),
            file_type=result.get("type"),
        )

        # Defer the heavy indexing (chunk + embed + store) to after the response,
        # so the user gets an instant reply. The frontend polls
        # GET /api/documents/{id}/status until status flips to "ready"/"failed".
        background_tasks.add_task(
            index_document,
            document.id,
            current_user.email,
            result.get("text", ""),
            session_id,
        )

        return {
            "document_id": document.id,
            "filename": result.get("filename"),
            "type": "document",
            "status": "processing",
        }

    # Images need no indexing — already done.
    return {
        "filename": result.get("filename"),
        "type": result.get("type"),
        "status": "uploaded",
    }


@router.get("/session/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: TokenData = Depends(get_current_user)):
    session_messages = await get_session(session_id, current_user.email)
    return {"session_id": session_id, "messages": session_messages}

@router.post("/reset/{session_id}")
async def reset_session(session_id: str, current_user: TokenData = Depends(get_current_user)):
    await clear_session(session_id, current_user.email)
    return {"status": "ok", "message": f"Session {session_id} cleared"}
