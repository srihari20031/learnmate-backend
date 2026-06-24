from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from typing import Optional


class AttachmentInput(BaseModel):
    filename: str
    type: str
    mime_type: str
    base64: Optional[str] = None


class AttachmentRef(BaseModel):
    id: str
    filename: str
    type: str
    mime_type: str
    base64: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    attachments: list[AttachmentInput] = []


class ChatMessage(BaseModel):
    role: str
    content: str
    sent_at: Optional[datetime] = None
    attachments: list[AttachmentRef] = []


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = "chatting"
    notion_urls: list[str] = []


class ChatCreateRequest(BaseModel):
    title: Optional[str] = "New Chat"


class ChatCreateResponse(BaseModel):
    chat_id: str
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatUpdateTitleRequest(BaseModel):
    title: str


class ChatListResponse(BaseModel):
    chats: list[ChatCreateResponse]


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]

    
class LearnRequest(BaseModel):
    message: str
    
class LearnResponse(BaseModel):
    response: str
    session_id: str
    status: str  # "chatting" or "completed"
    notion_urls: list[str] = []

class DocumentStatus(str, Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    uploaded_at: datetime