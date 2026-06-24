from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.notion import router as notion_router
from app.api.notion_auth import router as notion_auth_router
from app.api.learn import router as learn_router
from app.api.auth import router as auth_router
from app.core.qdrant import create_collection_if_not_exists
from app.database import close_db
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://learnmate-frontend-three.vercel.app"],  # your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],  # allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],  # allows Content-Type, session-id, authorization, etc.
)

app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(chat_router, prefix="/api/chat")
app.include_router(notion_router, prefix="/api/notion")
app.include_router(notion_auth_router, prefix="/api/notion", tags=["notion-auth"])
app.include_router(learn_router, prefix="/api/learn")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Learning Agent is running!"}

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()

@app.on_event("startup")
async def startup_event():
    create_collection_if_not_exists()