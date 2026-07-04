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


class Source(BaseModel):
    # A retrieved chunk that informed the answer — the "citation" the UI shows.
    id: int
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    filename: Optional[str] = None
    snippet: str
    cited: bool = False  # did the model actually reference this source ([n]) in its answer?


class NotionPage(BaseModel):
    title: str
    url: str


class ChatMessage(BaseModel):
    role: str
    content: str
    sent_at: Optional[datetime] = None
    attachments: list[AttachmentRef] = []
    # Set only on assistant messages. Optional/nullable so older stored messages
    # (saved before these existed) still deserialize. Persisted so the Notion
    # links and citations survive a page reload.
    notion_urls: Optional[list[str]] = None
    notion_pages: Optional[list[NotionPage]] = None
    sources: Optional[list[Source]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = "chatting"
    notion_urls: list[str] = []
    notion_pages: list[NotionPage] = []
    sources: list[Source] = []


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
    notion_pages: list[NotionPage] = []
    sources: list[Source] = []

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