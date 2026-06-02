from groq import Groq
from app.core.config import settings
from app.prompts.notes import NOTE_PROMPT

client = Groq(api_key=settings.groq_api_key)
MODEL = "llama-3.3-70b-versatile"

async def generate_note(topic: str, known_stack: str, target_tech: str) -> str:
    prompt = NOTE_PROMPT.format(
        topic=topic,
        known_stack=known_stack,
        target_tech=target_tech
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()