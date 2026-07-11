# app/services/tutor_service.py
#
# The mode-aware turn orchestrator. A chat is either in "chatting" mode (normal
# conversational tutor) or "teaching" mode (a guided, topic-by-topic walkthrough).
# handle_turn() is the single entry point the /message endpoint calls; it reads
# the session's mode from context and routes accordingly, intercepting the
# control tokens the model emits:
#
#   START_TEACHING     (chatting)  -> build a curriculum, enter teaching mode,
#                                     deliver topic 1
#   NEXT_TOPIC         (teaching)  -> save a personalized note for the current
#                                     topic (from the learner's doubts), advance
#   READY_TO_GENERATE  (chatting)  -> generate the full note set in one go
#
# State lives in the session context (contexts_collection):
#   mode, curriculum (list[str]), current_topic (int), current_topic_doubts (list)

import re

from app.core.session import get_context, set_context, add_message, get_session
from app.services.claude_service import (
    apply_sliding_window,
    to_llm_messages,
    build_llm_messages,
    retrieve_context_and_sources,
    mark_cited_sources,
    resolve_known_stack,
    extract_context,
    _maybe_extract_context,
    generate_curriculum_ai,
)
from app.prompts.intake import TEACHING_PROMPT
from app.services.llm import ai_invoke

START_TEACHING = "START_TEACHING"
NEXT_TOPIC = "NEXT_TOPIC"
READY_TO_GENERATE = "READY_TO_GENERATE"
CONTROL_TOKENS = [START_TEACHING, NEXT_TOPIC, READY_TO_GENERATE]


def strip_control_tokens(text: str) -> str:
    for token in CONTROL_TOKENS:
        text = re.sub(token, "", text or "", flags=re.IGNORECASE)
    return text.strip()


def _reply_is_token(reply: str, token: str) -> bool:
    # True only when the WHOLE reply is essentially just this control token
    # (ignoring punctuation/formatting). The prompts tell the model to emit a
    # control token "and nothing else", so this strict check lets us intercept a
    # genuine signal while REFUSING to act on a token that merely appears inside
    # prose — e.g. the model echoing a token name, or an uploaded document that
    # mentions "START_TEACHING" to hijack the control flow (a prompt-injection
    # vector). Substring matching here is exploitable; equality is not.
    cleaned = re.sub(r"[^A-Za-z_]", "", (reply or "")).upper()
    return cleaned == token


def is_advance_signal(reply: str) -> bool:
    return _reply_is_token(reply, NEXT_TOPIC)


def _result(reply, sources=None, notion_pages=None, status="chatting"):
    return {
        "reply": reply,
        "sources": sources or [],
        "notion_pages": notion_pages or [],
        "status": status,
    }


async def _complete(system_prompt: str, history: list) -> str:
    # One-shot completion with an explicit system prompt (used for the teaching
    # calls, which swap in the teaching prompt instead of the intake prompt).
    messages = [{"role": "system", "content": system_prompt}] + apply_sliding_window(
        to_llm_messages(history)
    )
    response = await ai_invoke(messages=messages, temperature=0.0)
    return response


def _build_teaching_prompt(ctx: dict) -> str:
    curriculum = ctx.get("curriculum", []) or []
    i = ctx.get("current_topic", 0)
    i = max(0, min(i, len(curriculum) - 1)) if curriculum else 0
    curriculum_list = "\n".join(f"{n + 1}. {t}" for n, t in enumerate(curriculum))
    return TEACHING_PROMPT.format(
        target_tech=ctx.get("target_tech", "the target technology"),
        known_stack=ctx.get("known_stack") or "a general programming background",
        total=len(curriculum),
        curriculum_list=curriculum_list,
        position=i + 1,
        current_topic=curriculum[i] if curriculum else "",
    )


async def _deliver_current_topic(session_id: str, user_email: str) -> str:
    # Produce the lesson text for whatever topic is now current. This is a fresh
    # DELIVERY, not a reply-handler: the learner's previous message was often
    # "next topic" / "understood", which the normal teaching prompt would read as
    # an advance signal and answer with NEXT_TOPIC — yielding an empty lesson
    # after stripping. So we explicitly tell the model to teach the topic now.
    ctx = await get_context(session_id, user_email)
    history = await get_session(session_id, user_email)
    system = _build_teaching_prompt(ctx) + (
        "\n\n## Deliver this topic now\n"
        "The learner has just arrived at THIS topic. Teach it now: a few short "
        "sections with a concrete example grounded in what they already know, "
        "ending by inviting doubts or an 'understood' to continue. You are "
        "TEACHING in this message — do NOT output NEXT_TOPIC here, regardless of "
        "what the learner's previous message said."
    )
    lesson = await _complete(system, history)
    return strip_control_tokens(lesson)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

async def handle_turn(
    session_id: str,
    user_message: str,
    user_email: str,
    attachments: list[dict] | None = None,
) -> dict:
    await add_message(
        session_id, "user", user_message, user_email, attachments=attachments
    )
    ctx = await get_context(session_id, user_email)

    if ctx.get("mode") == "teaching":
        history = await get_session(session_id, user_email)
        return await _handle_teaching(session_id, user_email, user_message, ctx, history)

    return await _handle_chatting(session_id, user_email, user_message, ctx)


# --------------------------------------------------------------------------
# chatting mode
# --------------------------------------------------------------------------

async def _handle_chatting(session_id, user_email, user_message, ctx) -> dict:
    document_context, sources = await retrieve_context_and_sources(
        session_id, user_email, user_message
    )
    history = await get_session(session_id, user_email)
    known_stack = await resolve_known_stack(session_id, user_email)
    messages = build_llm_messages(history, document_context, known_stack)

    reply = (
        await ai_invoke(messages=messages, temperature=0.0)
    )

    # Strict token interception: the whole reply must BE the token. A substring
    # check is exploitable — an uploaded document that merely names a token, or
    # the model echoing one in prose, could otherwise hijack the flow (seen in the
    # prompt-injection test, where naming START_TEACHING derailed a summarize
    # request into the teaching path).
    if _reply_is_token(reply, START_TEACHING):
        return await _start_teaching(session_id, user_email, sources)

    if _reply_is_token(reply, READY_TO_GENERATE):
        return await _generate_all_notes(session_id, user_email, sources)

    # plain conversational turn
    mark_cited_sources(reply, sources)
    display = strip_control_tokens(reply)
    await add_message(
        session_id, "assistant", display, user_email, sources=sources
    )
    await _maybe_extract_context(session_id, history, user_email)
    return _result(display, sources=sources, status="chatting")


async def _start_teaching(session_id, user_email, sources) -> dict:
    # Make sure target_tech/known_stack are captured before building a curriculum
    # (the user may have said "teach me X" in the very message that triggered this).
    history = await get_session(session_id, user_email)
    await extract_context(session_id, history, user_email)

    ctx = await get_context(session_id, user_email)
    target_tech = ctx.get("target_tech")
    if not target_tech:
        msg = "I'd love to walk you through it! Which technology should we start with?"
        await add_message(session_id, "assistant", msg, user_email)
        return _result(msg, sources=sources, status="chatting")

    known_stack = await resolve_known_stack(session_id, user_email) or "beginner"
    curriculum = await generate_curriculum_ai(known_stack, target_tech)

    await set_context(session_id, "curriculum", curriculum, user_email)
    await set_context(session_id, "current_topic", 0, user_email)
    await set_context(session_id, "current_topic_doubts", [], user_email)
    await set_context(session_id, "known_stack", known_stack, user_email)
    await set_context(session_id, "mode", "teaching", user_email)

    lesson = await _deliver_current_topic(session_id, user_email)
    intro = (
        f"Great — let's go through **{target_tech}** step by step. "
        f"We'll cover {len(curriculum)} topics, and I'll save a personalized note "
        f"to your Notion after each one.\n\n"
        f"**Topic 1 of {len(curriculum)}: {curriculum[0]}**\n\n"
    )
    reply = intro + lesson
    await add_message(session_id, "assistant", reply, user_email)
    return _result(reply, sources=sources, status="teaching")


async def _generate_all_notes(session_id, user_email, sources) -> dict:
    from app.api.learn import generate_learning_notes

    result = await generate_learning_notes(session_id, user_email)
    if result.get("notion_urls"):
        msg = "Your notes are ready in Notion!"
        await add_message(
            session_id,
            "assistant",
            msg,
            user_email,
            notion_urls=result["notion_urls"],
            notion_pages=result["notion_pages"],
        )
        return _result(
            msg,
            sources=sources,
            notion_pages=result["notion_pages"],
            status=result["status"],
        )

    msg = "I'd be happy to create notes! Which technology should I build them around?"
    await add_message(session_id, "assistant", msg, user_email)
    return _result(msg, sources=sources, status="chatting")


# --------------------------------------------------------------------------
# teaching mode
# --------------------------------------------------------------------------

async def _handle_teaching(session_id, user_email, user_message, ctx, history) -> dict:
    reply = await _complete(_build_teaching_prompt(ctx), history)

    if is_advance_signal(reply):
        return await _advance_topic(session_id, user_email, ctx)

    # A doubt / question — log it for this topic's note, and answer it.
    doubts = list(ctx.get("current_topic_doubts", []) or [])
    doubts.append(user_message)
    await set_context(session_id, "current_topic_doubts", doubts, user_email)

    display = strip_control_tokens(reply)
    await add_message(session_id, "assistant", display, user_email)
    return _result(display, status="teaching")


async def _advance_topic(session_id, user_email, ctx) -> dict:
    from app.api.learn import save_single_note

    curriculum = ctx.get("curriculum", []) or []
    i = ctx.get("current_topic", 0)
    target_tech = ctx.get("target_tech")
    known_stack = ctx.get("known_stack") or "beginner"
    doubts = ctx.get("current_topic_doubts", []) or []

    # Save the personalized note for the topic the learner just completed.
    page = None
    if 0 <= i < len(curriculum):
        try:
            page = await save_single_note(
                session_id, user_email, curriculum[i], known_stack, target_tech, doubts
            )
        except Exception as e:  # never let a Notion hiccup break the lesson flow
            print(f"[Tutor] Failed to save note for '{curriculum[i]}': {e}")

    next_i = i + 1
    await set_context(session_id, "current_topic", next_i, user_email)
    await set_context(session_id, "current_topic_doubts", [], user_email)

    pages = [page] if page else []
    urls = [page["url"]] if page else None
    saved_line = (
        f"✅ Your personalized notes on **{curriculum[i]}** are saved to Notion — "
        f"open the card below to review or edit them anytime.\n\n"
        if page
        else ""
    )

    # Finished the whole curriculum?
    if next_i >= len(curriculum):
        await set_context(session_id, "mode", "chatting", user_email)
        msg = (
            f"{saved_line}That was the last topic — you've completed the "
            f"**{target_tech}** track! 🎉 Every topic's note is saved in your Notion. "
            f"Want to revisit anything or learn something new?"
        )
        await add_message(
            session_id, "assistant", msg, user_email,
            notion_pages=pages or None, notion_urls=urls,
        )
        return _result(msg, notion_pages=pages, status="completed")

    # Otherwise deliver the next topic.
    lesson = await _deliver_current_topic(session_id, user_email)
    header = (
        f"{saved_line}"
        f"**Topic {next_i + 1} of {len(curriculum)}: {curriculum[next_i]}**\n\n"
    )
    reply = header + lesson
    await add_message(
        session_id, "assistant", reply, user_email,
        notion_pages=pages or None, notion_urls=urls,
    )
    return _result(reply, notion_pages=pages, status="teaching")
