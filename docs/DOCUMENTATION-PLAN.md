# LearnMate — Final Documentation Plan (READ THIS WHEN THE PROJECT IS DONE)

> **Status: PENDING — do not generate yet.** This is a marker/spec. When the user
> says LearnMate feature work is complete, generate the full documentation from
> this outline as a **PDF** (see "How to generate" at the bottom). Until then,
> keep this file updated as new features land so the outline stays current.

## What the user wants

A **thorough "interview-explainer book"** for the whole LearnMate backend — the kind
of document you'd study to confidently explain the system in an interview. For
**every** major concept and feature we built:

1. **What it is** — plain-language explanation of the concept, from scratch.
2. **Why we did it / why this choice** — the problem it solves, alternatives considered, the trade-off.
3. **How it works** — the actual mechanism, with **real code examples pulled from this codebase** (cite `file:line`).
4. **Gotchas / bugs we hit** — the mistakes and what they taught (these make the best interview stories).
5. **Interview Q&A** — 2–4 likely interview questions per topic with crisp answers.

Audience: the user (a full-stack dev moving into AI engineering) will **read it and
verify the code against the codebase**, so accuracy and real `file:line` references matter.

Tone: clear, concrete, teacherly. Prefer real snippets over pseudocode. Explain the
"why" behind every decision — that's what interviewers probe.

## Chapter outline (update as features are added)

> Mark each ✅ built / 🚧 in progress / ⬜ planned so the final gen knows what to cover.

### 1. System overview & architecture ✅
- Stack: FastAPI, MongoDB (motor), Qdrant, Groq (llama-3.3-70b), Notion, sentence-transformers.
- Request lifecycle; where each piece fits. A diagram-in-words.

### 2. The RAG pipeline ✅
- **Chunking** — `chunk_service.py`; the critical 256-token truncation bug (chunk_size must match the embedding model's max tokens).
- **Embeddings** — all-MiniLM-L6-v2 (384-dim); batch encoding; `asyncio.to_thread` offload (why CPU work must leave the event loop).
- **Vector store** — Qdrant, `AsyncQdrantClient`, cosine, collection setup on startup.
- **Dense retrieval** — `search_embeddings`, metadata filter (`user_email` + `session_id`) → multi-tenancy.
- **Hybrid search** — BM25 (`keyword_service.py`) + Reciprocal Rank Fusion (`hybrid_service.py`); why lexical + semantic beats either alone; the LRU cache for the BM25 index and its signature.
- **Reranking** — cross-encoder (`rerank_service.py`); recall-vs-precision (wide pool → rerank down).
- **Metadata filtering & context compression** — how/what we store; what "compression" means here.

### 3. Generation ✅
- Groq `AsyncGroq`; message assembly; numbered sources.
- **Citations** — `[n]` markers, `mark_cited_sources`, why we don't strip markers (code tool).
- **Grounding guardrail** — refuse on irrelevant context, not just missing answers.
- **Prompt-injection defense** — per-request random fence (`secrets.token_hex`); why an unforgeable delimiter stops break-out.

### 4. Async & concurrency ✅
- Event loop basics; `asyncio.to_thread` vs blocking; async I/O everywhere (motor/Qdrant/Groq/Redis).
- **FastAPI BackgroundTasks** — deferred document indexing; async task on loop vs sync task in threadpool; 202 + status polling.

### 5. SSE streaming ✅
- `StreamingResponse`, `text/event-stream`, event types, sentinel tail-buffer so control tokens never leak.

### 6. Sessions, persistence & Notion link storage ✅
- chats / messages / `session_contexts`; `add_message` extras (notion_urls/pages/sources); `update_last_message`.
- The "links null after restart" debugging story (stale server).

### 7. The profile lane (résumé → known_stack) ✅
- Why a résumé is a **profile, not RAG** (retrieval pollution, wasted storage, guard conflict).
- `profile_service.py`: classify + extract, user-wide storage, `merge_stacks`, `resolve_known_stack` (profile ∪ session).
- Endpoints: `POST/GET/DELETE /api/profile`.

### 8. Conversational tutor & guided teaching ✅
> Draft already written: `docs/system-prompt-design.md` — fold it in as this chapter (prompt design, control tokens, explicit-state injection, the failure→fix stories).
- **Opt-in notes** — the funnel bug (auto-generating instead of answering); `READY_TO_GENERATE` only on explicit request.
- **Guided topic-by-topic flow** — `tutor_service.py` mode-aware `handle_turn`; state (mode/curriculum/current_topic/doubts); `START_TEACHING` / `NEXT_TOPIC` control tokens; why explicit state beats prompt-only (sliding window).
- **Personalized notes from doubts** — `generate_personalized_note`; incremental save on "understood".
- **Category-aware comparisons** — same-category → side-by-side code; cross-category (Docker vs app-dev) → relate via workflow, no forced comparison.
- **Curriculum ordering** — fundamentals-first learning path.
- Bugs found & fixed via a 22-doubt stress test: false "notes ready" guard; **false-advance on a doubt** — a genuine question sometimes tripped `NEXT_TOPIC` (naive substring match), cutting a topic short and saving an empty non-personalized note. Fixed with strict token detection (`is_advance_signal`: whole reply must be just the token) + prompt reinforcement ("a question is never an advance signal"). Also fixed note **stack-recitation** (NOTE_PROMPT header dumped all ~20 techs → now picks the 1–2 relevant ones, category-aware). Verify live once the Groq daily token budget resets.

### 9. Notion integration ✅
- Markdown → Notion blocks; code-language normalization; parent-page-per-user.

### 10. Caching ✅
- Upstash Redis (async) for curriculum/notes; in-process LRU for BM25; when to cache vs not (personalized notes bypass cache).

### 11. Evaluation ✅
- Retrieval eval (hit@k, MRR); generation eval (faithfulness / refusal / injection) via LLM-as-judge.

### 12. Testing ✅
- pytest suite (keyword/hybrid/chunk/claude-helpers/profile); what each guards; a test that caught a wrong assumption.

### 13. Document processing ✅
- pypdf/docx/txt extraction; the deliberate choice of text-only over Claude PDF-vision for a "what to learn" use case.

### 14. Known limitations & the road to LangChain 🚧
- Streaming teaching not wired; BackgroundTasks not durable across restart; false-advance fix; the planned LangChain/LangGraph migration (RAG-as-LCEL → StateGraph → tool calling) and *why* (map hand-rolled state → framework).

## How to generate (when the user says "done")

1. Write the book as a single self-contained HTML file (reuse the print styling from
   the existing `docs/*.pdf` generators: cover page, section cards, code blocks with
   monospace, page-break-inside avoid).
2. Pull **real code snippets** from the codebase at generation time (don't paraphrase);
   include `file:line` references the user can click/verify.
3. Render to `docs/learnmate-explainer.pdf` via headless Chrome:
   `chrome --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=<out> file:///<html>`
   (Chrome path: `C:\Program Files\Google\Chrome\Application\chrome.exe`).
4. **Delete the temporary HTML** after the PDF is created (per the user's standing preference).
5. Keep it thorough — this is a study/interview artifact, length is fine.
