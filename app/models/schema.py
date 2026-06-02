from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    
class LearnRequest(BaseModel):
    message: str
    
class LearnResponse(BaseModel):
    response: str
    session_id: str
    status: str  # "chatting" or "completed"
    notion_urls: list[str] = []