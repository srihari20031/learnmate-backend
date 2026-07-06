from groq import AsyncGroq
from app.core.config import settings
from app.prompts.notes import NOTE_PROMPT

client = AsyncGroq(api_key=settings.groq_api_key)
MODEL = "llama-3.3-70b-versatile"

async def generate_note(topic: str, known_stack: str, target_tech: str) -> str:
    prompt = NOTE_PROMPT.format(
        topic=topic,
        known_stack=known_stack,
        target_tech=target_tech
    )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


async def generate_personalized_note(
    topic: str, known_stack: str, target_tech: str, doubts: list[str]
) -> str:
    # Same structure as generate_note, but tailored to the exact questions the
    # learner asked while going through this topic in the guided session. This is
    # the payoff of the topic-by-topic flow: the saved note captures what THEY
    # got stuck on, not just a generic explanation. Not cached — every learner's
    # doubts differ, so each personalized note is unique.
    prompt = NOTE_PROMPT.format(
        topic=topic,
        known_stack=known_stack,
        target_tech=target_tech,
    )
    if doubts:
        doubt_lines = "\n".join(f"- {d}" for d in doubts)
        prompt += (
            "\n\nWhile studying this topic, the learner specifically asked about "
            "the following. Make sure the note directly and clearly addresses each "
            "of these, adding a short '## Your Questions' section at the end that "
            "answers them:\n"
            f"{doubt_lines}\n"
        )
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()