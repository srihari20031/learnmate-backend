import sys

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Learning Agent"

    # LLM
    groq_api_key: str

    # Datastores
    mongodb_uri: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    upstash_redis_url: str
    upstash_redis_token: str

    # Notion
    notion_api_key: str
    notion_parent_page_id: str
    NOTION_CLIENT_ID: str
    NOTION_CLIENT_SECRET: str

    # Auth
    jwt_secret_key: str
    access_token_expire_minutes: int = 30

    # Misc
    FRONTEND_URL: str = "http://localhost:3000"
    chat_sliding_window_messages: int = 12


try:
    settings = Settings()
except ValidationError as exc:
    # Pydantic's ValidationError repr embeds `input_value` -- the whole dict of settings
    # it DID receive, secrets included. Letting it reach stdout writes every API key into
    # the platform's log store. So report only the field NAMES and re-raise with
    # `from None`, which drops the original exception (and its values) from the traceback.
    missing, invalid = [], []
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else "<unknown>"
        if err["type"] == "missing":
            missing.append(field)
        else:
            invalid.append(f"{field} ({err['msg']})")

    lines = ["Configuration error: the app cannot start."]
    if missing:
        lines.append(f"  Missing environment variables: {', '.join(sorted(missing))}")
    if invalid:
        lines.append(f"  Invalid environment variables: {', '.join(sorted(invalid))}")
    lines.append("  Set them in your deployment environment (or .env for local runs).")

    message = "\n".join(lines)
    print(message, file=sys.stderr)
    raise RuntimeError(message) from None
