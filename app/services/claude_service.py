from groq import Groq
from app.core.config import settings
from app.core.session import get_session, add_message, set_context, get_context
from app.prompts.intake import SYSTEM_PROMPT
import json

client = Groq(api_key=settings.groq_api_key)
MODEL = "llama-3.3-70b-versatile"

async def extract_context(session_id: str, history: list, current_user_email: str):
    prompt = f"""From this conversation extract:
1. target_tech - what the user wants to learn
2. known_stack - what they already know

Conversation: {history}

Return ONLY JSON like this, nothing else:
{{"target_tech": "FastAPI", "known_stack": "Node.js"}}
If not found return null for that field.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        print("Extracted context:", data)
        if data.get("target_tech"):
            await set_context(session_id, "target_tech", data["target_tech"], current_user_email)
        if data.get("known_stack"):
            await set_context(session_id, "known_stack", data["known_stack"], current_user_email)
    except Exception as e:
        print("Context extraction failed:", e)


async def get_claude_response(session_id: str, user_message: str, current_user_email: str) -> str:
    await add_message(session_id, "user", user_message, current_user_email)
    history = await get_session(session_id, current_user_email)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    reply = response.choices[0].message.content
    await add_message(session_id, "assistant", reply, current_user_email)

    context = await get_context(session_id, current_user_email)
    if not context.get("target_tech") or not context.get("known_stack"):
        await extract_context(session_id, history, current_user_email)

    print("Current context:", await get_context(session_id, current_user_email))
    return reply


async def generate_curriculum_ai(known_stack: str, target_tech: str) -> list[str]:
    prompt = f"""Generate a list of 6-8 important topics to learn for {target_tech} 
for a developer who already knows {known_stack}.

Return ONLY a JSON array of topic strings, nothing else.
Example: ["Routing", "Middleware", "Authentication"]
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    topics = json.loads(text)
    return topics