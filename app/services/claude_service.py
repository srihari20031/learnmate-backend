import asyncio
import re
from types import SimpleNamespace
from groq import AsyncGroq
from app.core.config import settings
from app.core.session import get_session, add_message, set_context, get_context
from app.prompts.intake import SYSTEM_PROMPT
import json

client = AsyncGroq(api_key=settings.groq_api_key)
MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHARS = 100_000


def apply_sliding_window(history: list) -> list:
    # MongoDB still stores the full chat history, but sending the full history
    # to the LLM on every request can exceed token limits and increase cost.
    window_size = max(settings.chat_sliding_window_messages, 2)

    # Keep an even number of messages so the model receives complete
    # user/assistant turns instead of ending in the middle of a conversation.
    if window_size % 2 != 0:
        window_size -= 1

    return history[-window_size:]


def to_llm_messages(history: list) -> list:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in history
        if message.get("role") and message.get("content") is not None
    ]


async def extract_context(session_id: str, history: list, current_user_email: str):
    # Context extraction also uses the sliding window so long conversations do
    # not create oversized prompts when extracting target_tech and known_stack.
    recent_history = apply_sliding_window(to_llm_messages(history))
    prompt = f"""From this conversation extract:
1. target_tech - what the user wants to learn
2. known_stack - what they already know

Conversation: {recent_history}

Return ONLY JSON like this, nothing else:
{{"target_tech": "FastAPI", "known_stack": "Node.js"}}
If not found return null for that field.
"""
    response = await client.chat.completions.create(
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


async def get_claude_response(
    session_id: str,
    user_message: str,
    current_user_email: str,
    attachments: list[dict] | None = None,
) -> str:
    from app.services.embedding import generate_embeddings, search_embeddings
    from app.services.rerank_service import rerank
    from app.services.rag_service import get_session_chunks
    from app.services.hybrid_service import hybrid_rank

    # Hybrid retrieval, three stages:
    #   recall    — dense (vector) + sparse (BM25), each casting a wide net
    #   fusion    — Reciprocal Rank Fusion merges the two id lists into one pool
    #   precision — the cross-encoder reranks that pool down to the final few
    CANDIDATE_POOL = 20
    FINAL_TOP_K = 5

    document_context = ""

    # --------------------------------------------------
    # Fetch the session's chunks once. This doubles as the retrieval gate
    # (empty -> skip everything) and as the BM25 corpus + id->text map below,
    # so a plain conversation turn costs just this one indexed Mongo read.
    # --------------------------------------------------

    session_chunks = await get_session_chunks(current_user_email, session_id)

    if session_chunks:

        # Dense recall. encode() is CPU-bound -> worker thread; the Qdrant query
        # uses the async client, so it's a direct await (native non-blocking I/O).
        query_embedding = await asyncio.to_thread(generate_embeddings, user_message)
        dense_points = await search_embeddings(
            query_vector=query_embedding,
            user_email=current_user_email,
            session_id=session_id,
            top_k=CANDIDATE_POOL,
        )
        dense_ids = [
            p.payload.get("chunk_id")
            for p in dense_points
            if p.payload.get("chunk_id")
        ]

        # Sparse (BM25) recall + RRF fusion. Building the BM25 index over the
        # session corpus is CPU work, so it's offloaded to a worker thread.
        fused_ids = await asyncio.to_thread(
            hybrid_rank,
            user_message,
            session_chunks,
            dense_ids,
            CANDIDATE_POOL,
            session_id=session_id,
        )

        # Turn fused ids back into rerank-compatible points (.score + .payload).
        text_by_id = {c["chunk_id"]: c["text"] for c in session_chunks}
        candidates = [
            SimpleNamespace(
                score=0.0,
                payload={"text": text_by_id[cid], "chunk_id": cid},
            )
            for cid in fused_ids
            if cid in text_by_id
        ]

        # Cross-encoder precision pass over the fused pool (CPU-bound, offloaded).
        reranked_results = await asyncio.to_thread(
            rerank, user_message, candidates, FINAL_TOP_K
        )

        retrieved_chunks = [
            result.payload.get("text")
            for result in reranked_results
            if result.payload.get("text")
        ]

        document_context = "\n\n".join(retrieved_chunks)

    # --------------------------------------------------
    # Save user message
    # --------------------------------------------------

    await add_message(
        session_id,
        "user",
        user_message,
        current_user_email,
        attachments=attachments,
    )

    history = await get_session(
        session_id,
        current_user_email,
    )

    recent_history = apply_sliding_window(to_llm_messages(history))

    # --------------------------------------------------
    # Build system prompt
    # --------------------------------------------------

    system_prompt = SYSTEM_PROMPT

    if document_context:
        system_prompt += (
            "\n\n--- Retrieved Document Context ---\n"
            + document_context[:MAX_CONTEXT_CHARS]
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + recent_history

    # --------------------------------------------------
    # Call LLM
    # --------------------------------------------------

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    reply = response.choices[0].message.content

    # --------------------------------------------------
    # Save assistant reply
    #
    # READY_TO_GENERATE is an internal control sentinel (see intake prompt),
    # not user-facing text. Strip it from what we persist/display so it never
    # leaks into the chat bubble on history reload. We still RETURN the raw
    # reply so the caller (chat.py) can detect the sentinel and trigger note
    # generation.
    # --------------------------------------------------

    display_reply = re.sub(
        r"READY_TO_GENERATE", "", reply, flags=re.IGNORECASE
    ).strip()

    await add_message(
        session_id,
        "assistant",
        display_reply,
        current_user_email,
    )

    # --------------------------------------------------
    # Existing curriculum extraction logic
    # --------------------------------------------------

    processed_context = await get_context(
        session_id,
        current_user_email,
    )

    if (
        not processed_context.get("target_tech")
        or not processed_context.get("known_stack")
    ):
        await extract_context(
            session_id,
            history,
            current_user_email,
        )

    return reply


async def generate_curriculum_ai(known_stack: str, target_tech: str) -> list[str]:
    prompt = f"""Generate a list of 6-8 important topics to learn for {target_tech} 
for a developer who already knows {known_stack}.

Return ONLY a JSON array of topic strings, nothing else.
Example: ["Routing", "Middleware", "Authentication"]
"""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    topics = json.loads(text)
    return topics