from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Learning Agent"
    claude_api_key: str
    notion_api_key: str
    notion_parent_page_id: str  
    openai_api_key: str
    gemini_api_key: str
    groq_api_key: str
    mongodb_uri: str
    access_token_expire_minutes: int = 30
    NOTION_CLIENT_ID: str
    NOTION_CLIENT_SECRET: str
    NOTION_REDIRECT_URI: str
    upstash_redis_url: str
    upstash_redis_token: str
    jwt_secret_key: str
    FRONTEND_URL: str = "http://localhost:3000"
    chat_sliding_window_messages: int = 12
    QDRANT_URL: str
    QDRANT_API_KEY: str
    

    class Config:
        env_file = ".env"

settings = Settings()