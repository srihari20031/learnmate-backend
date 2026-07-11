from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from app.core.config import settings

MODEL = "llama-3.3-70b-versatile"

def get_chat_model(temperature: float = 0.0, max_retries: int = 2, **kw) -> ChatGroq:
    # max_retries is a named param, NOT hard-coded before **kw: otherwise a caller
    # that also passes max_retries in kw raises "multiple values for keyword argument".
    return ChatGroq(model=MODEL, api_key=settings.groq_api_key,
                    temperature=temperature, max_retries=max_retries, **kw)

_ROLE = {"system": SystemMessage, "user": HumanMessage, "human": HumanMessage,
         "assistant": AIMessage, "ai": AIMessage}

def _to_messages(messages: list[dict]) -> list[BaseMessage]:
    out = []
    for m in messages:
        cls = _ROLE.get(m.get("role"))          # .get, so an unknown role doesn't KeyError
        if cls and m.get("content") is not None:
            out.append(cls(content=m["content"]))  # only content
    return out

async def ai_invoke(messages: list[dict], temperature: float = 0.0, **kw) -> str:
    chat_model = get_chat_model(temperature=temperature, **kw)
    msgs = _to_messages(messages)
    # _to_messages already returns a list[BaseMessage]; ainvoke takes that list
    # directly. Wrapping it again ([msgs]) makes a list-of-lists and fails.
    response = await chat_model.ainvoke(msgs)
    return response.content.strip()