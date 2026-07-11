# tests/conftest.py
#
# Test doubles for the turn orchestrator (app/services/tutor_service.py).
#
# Why this file exists
# -------------------
# `handle_turn` is the single code path every chat message flows through, and
# until now it had ZERO tests -- because exercising it meant a live Groq call, a
# live MongoDB, a live Qdrant, and a live Notion page. So nobody tested it, and
# the three bugs it has shipped (false NEXT_TOPIC advance, prompt-injection via a
# document that merely NAMED a control token, and a token leaking into the UI)
# were all found by hand, in production, by a human reading chat transcripts.
#
# The two doubles below remove every external dependency from that path:
#
#   FakeLLM    -- replaces the Groq client. You SCRIPT what the model says, which
#                 is the only way to test "what happens when the model emits
#                 exactly NEXT_TOPIC" versus "...mentions NEXT_TOPIC in prose".
#                 A real LLM cannot be made to do either on demand.
#
#   FakeStore  -- replaces the four MongoDB-backed session functions with dicts.
#
# Together they make the orchestrator's decision logic -- which is all that
# matters here -- testable in milliseconds, deterministically, and for zero Groq
# tokens (the free tier caps at 100k/day, and a single stress test once ate 96k).
#
# How the patching works
# ---------------------
# tutor_service does `from app.core.session import get_context, ...` at import
# time. That binds those names into `tutor_service`'s OWN module namespace. So we
# patch `tutor_service.get_context`, NOT `app.core.session.get_context` -- the
# latter would have no effect, because tutor_service already holds a reference to
# the original function object.
#
# The two `app.api.learn` functions are the exception: tutor_service imports them
# lazily INSIDE the function bodies (to break an import cycle), so the lookup
# happens at call time and we patch them at their source module.

import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# FakeLLM -- a scripted stand-in for the single LLM seam, app/services/llm.ai_invoke
# ---------------------------------------------------------------------------

class FakeLLM:
    """Replaces `ai_invoke(messages, **kw) -> str`, the one function every LLM
    call in the app now routes through after the ChatGroq migration (Phase 1).

    Scripting replies in order is what lets us drive the orchestrator down an
    exact branch. Running out of replies is a hard error, not a silent "" --
    an unexpected extra LLM call is itself a regression worth failing on.
    """

    def __init__(self):
        self.replies: list[str] = []
        self.calls: list[SimpleNamespace] = []

    def script(self, *replies: str) -> "FakeLLM":
        self.replies = list(replies)
        return self

    async def ai_invoke(self, messages=None, **kwargs) -> str:
        # Same signature ai_invoke exposes: messages (list[dict]) + kwargs like
        # temperature. Records the call so system_prompt() can inspect it.
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))

        if not self.replies:
            raise AssertionError(
                f"FakeLLM: the orchestrator made an unexpected extra LLM call "
                f"(call #{len(self.calls)}), but the script is exhausted.\n"
                f"Either the code under test changed how many times it calls the "
                f"model, or the test needs another scripted reply."
            )

        return self.replies.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def system_prompt(self, call_index: int = 0) -> str:
        """The system prompt sent on the Nth call. Lets a test assert on what the
        model was actually told (e.g. that the teaching prompt carried the right
        current topic) rather than only on what it replied."""
        messages = self.calls[call_index].messages
        return next((m["content"] for m in messages if m["role"] == "system"), "")


# ---------------------------------------------------------------------------
# FakeStore -- in-memory stand-in for the Mongo-backed session layer
# ---------------------------------------------------------------------------

class FakeStore:
    """Replaces get_context / set_context / add_message / get_session.

    `contexts` mirrors the `session_contexts.data` sub-document, which is where
    the guided-session state machine lives (mode, curriculum, current_topic,
    current_topic_doubts). Asserting on it directly is how we verify a state
    transition actually happened, instead of inferring it from the reply text.
    """

    def __init__(self, **initial_context):
        self.contexts: dict = dict(initial_context)
        self.messages: list[dict] = []

    async def get_context(self, session_id, current_user_email):
        # A copy, so a caller holding an old ctx can't observe later writes --
        # matching Mongo, where get_context returns a snapshot.
        return dict(self.contexts)

    async def set_context(self, session_id, key, value, current_user_email):
        self.contexts[key] = value

    async def add_message(
        self,
        session_id,
        role,
        content,
        current_user_email,
        attachments=None,
        notion_urls=None,
        notion_pages=None,
        sources=None,
    ):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "attachments": attachments,
                "notion_urls": notion_urls,
                "notion_pages": notion_pages,
                "sources": sources,
            }
        )

    async def get_session(self, session_id, current_user_email):
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    # -- convenience accessors for assertions --------------------------------

    @property
    def assistant_messages(self) -> list[dict]:
        return [m for m in self.messages if m["role"] == "assistant"]

    @property
    def last_assistant(self) -> dict:
        return self.assistant_messages[-1]


# ---------------------------------------------------------------------------
# Notion recorder -- captures what would have been written to a real page
# ---------------------------------------------------------------------------

class NotionRecorder:
    def __init__(self):
        self.saved_notes: list[dict] = []
        self.batch_calls: int = 0
        self.raise_on_save: Exception | None = None
        # What generate_learning_notes() should hand back. Default = success.
        self.batch_result: dict = {
            "status": "completed",
            "notion_urls": ["https://notion.so/page"],
            "notion_pages": [{"title": "Docker", "url": "https://notion.so/page"}],
        }

    async def save_single_note(
        self, session_id, user_email, topic, known_stack, target_tech, doubts=None
    ):
        if self.raise_on_save:
            raise self.raise_on_save
        self.saved_notes.append(
            {
                "topic": topic,
                "known_stack": known_stack,
                "target_tech": target_tech,
                "doubts": list(doubts or []),
            }
        )
        return {"title": topic, "url": f"https://notion.so/{len(self.saved_notes)}"}

    async def generate_learning_notes(self, session_id, user_email):
        self.batch_calls += 1
        return self.batch_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_CURRICULUM = [
    "Images and Containers",
    "Writing a Dockerfile",
    "Volumes and Data Persistence",
]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def notion():
    return NotionRecorder()


@pytest.fixture
def make_tutor(monkeypatch, fake_llm, notion):
    """Returns a factory: `make_tutor(**initial_context) -> (tutor_service, store)`.

    Imports happen inside the fixture, not at module import, so the pure-logic
    test modules still collect even if the app's settings/env are unavailable.
    """

    def _build(**initial_context):
        import app.services.tutor_service as tutor
        import app.api.learn as learn

        store = FakeStore(**initial_context)

        # The single LLM seam, bound into tutor_service's namespace by its
        # `from app.services.llm import ai_invoke`. Patch the bound name here,
        # not llm.ai_invoke -- tutor already holds its own reference.
        monkeypatch.setattr(tutor, "ai_invoke", fake_llm.ai_invoke)

        # The Mongo-backed session layer.
        monkeypatch.setattr(tutor, "get_context", store.get_context)
        monkeypatch.setattr(tutor, "set_context", store.set_context)
        monkeypatch.setattr(tutor, "add_message", store.add_message)
        monkeypatch.setattr(tutor, "get_session", store.get_session)

        # Retrieval: Qdrant + BM25 + the cross-encoder. Tested separately in
        # test_rerank_service.py / test_hybrid_service.py; here it's noise.
        # Default = a session with no documents.
        async def _no_documents(session_id, user_email, user_message):
            return "", []

        monkeypatch.setattr(tutor, "retrieve_context_and_sources", _no_documents)

        async def _known_stack(session_id, user_email):
            return "Python, FastAPI"

        monkeypatch.setattr(tutor, "resolve_known_stack", _known_stack)

        # extract_context and _maybe_extract_context each make their own LLM call
        # and write to Mongo. Stub them: extract_context is what discovers
        # target_tech, so it must still seed the store for _start_teaching.
        async def _extract_context(session_id, history, user_email):
            store.contexts.setdefault("target_tech", "Docker")

        async def _maybe_extract_context(session_id, history, user_email):
            return None

        monkeypatch.setattr(tutor, "extract_context", _extract_context)
        monkeypatch.setattr(tutor, "_maybe_extract_context", _maybe_extract_context)

        async def _curriculum(known_stack, target_tech):
            return list(DEFAULT_CURRICULUM)

        monkeypatch.setattr(tutor, "generate_curriculum_ai", _curriculum)

        # Imported lazily inside _advance_topic / _generate_all_notes, so the
        # lookup happens at call time -- patch them at the source module.
        monkeypatch.setattr(learn, "save_single_note", notion.save_single_note)
        monkeypatch.setattr(learn, "generate_learning_notes", notion.generate_learning_notes)

        return tutor, store

    return _build


@pytest.fixture
def teaching_context():
    """A session mid-way through a guided Docker session, on topic 1 of 3."""
    return {
        "mode": "teaching",
        "curriculum": list(DEFAULT_CURRICULUM),
        "current_topic": 0,
        "current_topic_doubts": [],
        "target_tech": "Docker",
        "known_stack": "Python, FastAPI",
    }
