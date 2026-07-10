# LearnMate → LangGraph Migration Plan

> **Status:** approved, not started. Living document — update the progress table as each phase lands.
> **Branch:** `langgraph-migration` (to be cut from `Langchain-integration` after the baseline is committed)

## Progress

| Phase | What | Status |
|---|---|---|
| — | Commit baseline, cut branch, rotate `jwt_secret_key` | ☐ not started |
| 0 | Safety net: `FakeLLM`, golden tests for `handle_turn`, rerank-floor tests | ☐ not started |
| 1 | Model layer: `ChatGroq` behind an `acomplete()` seam | ☐ not started |
| 2 | Native tool calling replaces the control tokens | ☐ not started |
| 3 | `StateGraph` replaces `handle_turn`'s branching; adds intent routing | ☐ not started |
| 4 | Restore true token streaming via `astream_events` | ☐ not started |

---

## Context

LearnMate works and is deployed, but the turn orchestrator in `app/services/tutor_service.py` is a hand-rolled state machine whose control plane is **string equality on control tokens** (`START_TEACHING`, `NEXT_TOPIC`, `READY_TO_GENERATE`). That design has already produced three real bugs this project fixed by hand:

1. A genuine learner question tripped `NEXT_TOPIC`, saving an empty note and skipping a topic.
2. An uploaded document that merely *named* `START_TEACHING` hijacked a summarize request (prompt injection).
3. The token leaked into the chat UI as raw text.

It also cannot do things the product needs: if a learner is mid-lesson on Rust and asks about FastAPI, there is no intent routing — the question is silently logged as a "doubt" on the Rust topic. Same for "actually I also know React", which should merge into `known_stack` but instead becomes a doubt.

**Goal:** replace the control-token state machine with a LangGraph `StateGraph` and native tool calling, while *keeping* the retrieval pipeline that was carefully tuned. Secondary goal: learn the LangChain/LangGraph abstractions that AI-engineer roles ask about.

**Non-goal:** rewriting retrieval into LCEL. See "What we deliberately do not migrate".

---

## Decisions already made

| Decision | Choice |
|---|---|
| Scope | LangGraph-first, minimal LangChain. Adopt `ChatGroq`, messages, `.bind_tools()`, `StateGraph`. Keep retrieval as-is. |
| Safety net | Build it **first** (Phase 0), before touching orchestration. |
| State persistence | Keep the existing Mongo `session_contexts` doc. No `langgraph-checkpoint-mongodb`. |

---

## Hard constraint discovered during planning

`langchain-groq` is capped at **1.1.3**, which requires `groq<1.0.0,>=0.30.0`.
The project currently pins **`groq==1.2.0`**.

**Adopting `ChatGroq` requires downgrading `groq` to `0.37.1`** (highest `<1.0.0`).

This is low-risk — every call site uses only `AsyncGroq(api_key=...)` → `.chat.completions.create(model=, messages=)` → `.choices[0].message.content`, which is stable across `0.37 → 1.x`. But it must be verified explicitly in Phase 1, not assumed. Once every call site is on `ChatGroq`, the direct `groq` pin is dropped entirely and `groq` becomes a transitive dependency of `langchain-groq`.

Package set to add:
```
langchain-core>=1.4.7,<2     # floor set by langgraph 1.2.8
langchain-groq==1.1.3
langgraph==1.2.8
groq==0.37.1                 # TEMPORARY: pinned down for langchain-groq; removed after Phase 2
```
Dev: `pytest-asyncio`.

---

## Phase 0 — Safety net (do this first)

**Why:** `tutor_service.py` has **zero test coverage**. No test imports it. There is no LLM mock anywhere in the project. `rerank_service.rerank` — including the relevance floor that fixed the "Azure PDF cited for a Redis question" bug — is also untested. Today we could rewrite `handle_turn` entirely, get a green suite, and not know we broke it.

Building the fakes first also means Phases 1–4 burn **almost no Groq tokens**, which matters against the 100k/day free-tier cap.

**Files to add:**

- `tests/conftest.py`
  - `FakeLLM` — scripted replies, records the messages it was called with.
  - `fake_store` — in-memory stand-in for `get_context` / `set_context` / `add_message` / `get_session`.
  - Both are installed by monkeypatching the names **in `tutor_service`'s namespace** (it does `from ... import client, get_context, ...`, so the module-level binding is what gets patched).

- `tests/test_tutor_service.py` — golden behavior tests pinning current semantics:
  - chatting → plain reply, sources attached, `status == "chatting"`
  - chatting + reply is exactly `START_TEACHING` → curriculum built, `mode == "teaching"`, topic 1 delivered, `status == "teaching"`
  - chatting + reply *contains but is not* `START_TEACHING` (prose or an injected document) → **must not** enter teaching (regression guard for the injection bug)
  - teaching + learner asks a question → `current_topic_doubts` grows, topic does **not** advance (regression guard for the false-advance bug)
  - teaching + reply is exactly `NEXT_TOPIC` → note saved with doubts, `current_topic` increments, next topic delivered
  - advancing past the last topic → `status == "completed"`, `mode` back to `"chatting"`
  - a Notion failure inside `_advance_topic` does not break the lesson flow

- `tests/test_rerank_service.py` — monkeypatch `reranker.predict` to return fixed logits (avoids loading the cross-encoder):
  - all-positive scores → up to `top_k` returned
  - mixed scores → sub-`0.0` chunks dropped, so fewer than `top_k` come back
  - all-negative scores → **exactly one** chunk survives (the keep-top-1 rule that keeps the grounding guard engaged)
  - empty candidates → `[]`

**Also add** `pytest-asyncio` to `requirements-dev.txt` and `asyncio_mode = auto` to `pytest.ini`.

**Verify:** `pytest` — expect the existing 44 plus roughly 12 new, all green. These tests must keep passing, unmodified, through every later phase. If a phase requires editing them, that phase is changing behavior and needs a conscious decision.

---

## Phase 1 — Model layer: `ChatGroq`

Introduce one seam so no other module talks to an LLM SDK directly.

- **New** `app/services/llm.py`:
  - `get_chat_model(temperature=0.0, **kw) -> BaseChatModel` → `ChatGroq(model="llama-3.3-70b-versatile", ...)`
  - `async def acomplete(system: str, history: list[dict]) -> str` — dict-history → `SystemMessage`/`HumanMessage`/`AIMessage`, invoke, return `.content`.

- **Rewrite call sites to use `acomplete`** (7 total — every `client.chat.completions.create` in the tree):
  - `claude_service.py`: `extract_context` (:56), `get_claude_response` (:288), `generate_curriculum_ai` (:373)
  - `tutor_service.py`: `_complete` (:79), `_handle_chatting` (:153)
  - `note_generator.py`: `generate_note` (:14), `generate_personalized_note` (:43)

  Confirm with `grep -rn "chat.completions.create" app/` — when Phase 1 is done this must return zero hits.

- Downgrade `groq` to `0.37.1`; **verify** `AsyncGroq` import and response shape still work before proceeding.

- Reuse unchanged: `apply_sliding_window`, `to_llm_messages`, `build_llm_messages`, `build_context_guard`, `build_known_stack_preamble`.

**Verify:** Phase-0 tests green with `FakeLLM` now standing in for `acomplete`. Then one real smoke turn against the live API.

---

## Phase 2 — Tool calling replaces control tokens

**New** `app/services/tools.py`, three tools with Pydantic arg schemas:

```
start_guided_session(technology: str)
advance_to_next_topic()
generate_all_notes(technology: str)
```

Bind with `.bind_tools([...])`. The model now returns an `AIMessage` carrying `tool_calls` instead of a magic string.

**Deleted:** `_reply_is_token`, `strip_control_tokens`, `CONTROL_TOKENS`, `SENTINEL`, `strip_sentinel`.
**Rewritten:** the control-token sections of `app/prompts/intake.py` (`SYSTEM_PROMPT` and `TEACHING_PROMPT`) — they describe tools, and the "NEVER reveal the internal control tokens" rule becomes unnecessary.

**Keep defense in depth.** Tool calling removes the *parsing* fragility and makes a document-that-names-a-token inert, but it does **not** make the model's intent judgement correct. So the graph still enforces preconditions deterministically:
- `start_guided_session` honored only when `mode == "chatting"`
- `advance_to_next_topic` honored only when `mode == "teaching"`

**Verify:** the Phase-0 injection and false-advance tests must still pass, rewritten only in how they *script* the fake model (tool call instead of token string) — never in what they assert.

---

## Phase 3 — `StateGraph` replaces `handle_turn`'s branching

**New package** `app/graph/`:

- `state.py` — `TutorState` TypedDict: `session_id, user_email, user_message, mode, curriculum, current_topic, current_topic_doubts, known_stack, target_tech, messages, sources, notion_pages, reply, status`
- `nodes.py` — one function per node:
  - `load_state` — rehydrate from `contexts_collection` (via existing `get_context`)
  - `retrieve` — calls existing `retrieve_context_and_sources` **unchanged**
  - `route_intent` — **the new capability.** In teaching mode, classify the learner message as `doubt | advance | switch_tech | state_background`
  - `chat`, `deliver_topic`, `advance_topic`, `generate_notes`
  - `persist` — write changed keys back via existing `set_context` / `add_message`
- `build.py` — `StateGraph(TutorState)` + conditional edges; compiled once at import.

`handle_turn(session_id, user_message, user_email, attachments)` survives as a **thin wrapper** that invokes the graph and returns the same `{reply, sources, notion_pages, status}` dict.

> This is the key to a safe migration: `app/api/chat.py`, `app/api/learn.py`, `get_claude_response_stream`, `ChatResponse`, `LearnResponse` and the **entire frontend contract stay byte-identical**. Only the internals of `handle_turn` change.

**State handling (per the chosen approach):** no LangGraph checkpointer. The graph is stateless across turns — `load_state` rehydrates from Mongo at entry, `persist` writes deltas at exit. Conversation history stays in `chats.messages[]` exactly as today. Nothing to migrate, and existing in-flight teaching sessions keep working.

**What `route_intent` fixes:**
- learning Rust, asks about FastAPI → answered as a side question, Rust lesson state untouched
- "I also know React" mid-lesson → merged into `known_stack` via the existing `merge_stacks`, not logged as a doubt

**Verify:** all Phase-0 golden tests still green, unmodified in their assertions. Add tests for the two `route_intent` behaviors above.

---

## Phase 4 — Restore true streaming

Today `get_claude_response_stream` runs the whole blocking turn, then *fakes* streaming by re-chunking the finished reply six words at a time (`_chunk_for_stream`). Real token streaming was lost when the streaming path was unified onto `handle_turn`.

Replace with `graph.astream_events(...)`, mapping LangGraph events onto the SSE envelope the frontend already consumes: `{type: "sources"}` → `{type: "delta"}`* → `{type: "done"}`.

**Verify:** frontend renders progressively during a chatting turn; a teaching turn still emits `status: "teaching"` and any `notion_pages`. `_chunk_for_stream` is deleted.

---

## What we deliberately do not migrate

This is the load-bearing part of the plan, and a good interview answer.

| Module | Why it stays |
|---|---|
| `rerank_service.py` | LangChain's `CrossEncoderReranker` takes `top_n` but **no score threshold**. Porting to it silently reintroduces the exact Azure-PDF-cited-for-a-Redis-question bug, because `top_k` would go back to being a quota instead of a cap. |
| `hybrid_service.py` | RRF at `k=60` with a tuned `pool_size`. `EnsembleRetriever` does RRF but buries the tuning behind defaults, and this module has real tests. |
| `keyword_service.py` | Hand-rolled BM25, dependency-free, tested, with a per-session cache keyed on the chunk-id signature. |
| `chunk_service.py` | 256-token chunking against the real MiniLM tokenizer. `evals/corpus.json` is frozen against its current boundaries. |
| `retrieve_context_and_sources` | The recall → RRF → rerank → numbered-context wiring, and the `[n]` ↔ source alignment the UI depends on. Wrapped by the `retrieve` node, not replaced. |

---

## Verification strategy

- **Per phase:** `pytest` must stay green *without editing the assertions* in `tests/test_tutor_service.py`. Assertion churn = behavior change = stop and decide.
- **Retrieval regression:** `python -m evals.run_eval` — hit@1/3/5 and MRR must not drop. Loads the local models, hits no LLM, costs no tokens.
- **Generation regression:** `python -m evals.eval_generation` — faithfulness / refusal / injection-resistance. **Hits the live Groq API**, so run once per phase, not per commit (100k tokens/day cap).
- **Manual:** the 16 scenarios in `docs/testing-scenarios.pdf` from the frontend. Scenario 9 (prompt injection) and Scenario 12 (category mismatch) are the ones this migration is most likely to disturb.

---

## Reference documentation

All links verified live during planning (July 2026). **Note:** LangChain moved its docs to `docs.langchain.com`; the old `python.langchain.com/docs/how_to/...` and `langchain-ai.github.io/langgraph/...` URLs now 308-redirect or are dead. Ignore older tutorials pointing there.

| Phase | Topic | Link |
|---|---|---|
| 1 | `ChatGroq` — install, import, `model=` param, async + tool-calling support matrix | https://docs.langchain.com/oss/python/integrations/chat/groq |
| 1 | Chat models: invoke/ainvoke, message types | https://docs.langchain.com/oss/python/langchain/models |
| 2 | Defining tools: `@tool`, docstring-as-description, Pydantic arg schemas | https://docs.langchain.com/oss/python/langchain/tools |
| 2 | `bind_tools()` and reading `response.tool_calls` — see the **"Tool calling"** section | https://docs.langchain.com/oss/python/langchain/models |
| 3 | `StateGraph`, State as TypedDict, `add_node` / `add_edge` / `add_conditional_edges` / `compile()` | https://docs.langchain.com/oss/python/langgraph/graph-api |
| 3 | Persistence, checkpointers, threads — background for *why we are not using one* | https://docs.langchain.com/oss/python/langgraph/persistence |

The single snippet that captures Phase 2, straight from the Models page:

```python
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    print(tool_call["name"], tool_call["args"])
```

That `response.tool_calls` list is precisely what replaces `_reply_is_token(reply, START_TEACHING)`.

---

## Sequencing and open items

Phases are ordered so each is independently shippable and revertible. We implement and discuss **one at a time**.

Before Phase 0:
- **Commit the current baseline.** Everything from the previous session (profile lane, guided teaching, rerank floor, document delete, `requirements.txt`/Dockerfile fixes, `config.py` hardening) is still uncommitted. Then branch `langgraph-migration`.
- **Rotate `jwt_secret_key`** — it was printed in full in the Railway crash logs.

Deferred, revisit after Phase 3:
- LangSmith tracing (needs an account; `ChatGroq` from Phase 1 makes it an env-var flip)
- A web-search tool — trivial once Phase 2 lands
- "Ask me to change the note" — a natural fourth tool
- A real `BaseCheckpointSaver` for time-travel/replay, if the concept is worth learning hands-on
