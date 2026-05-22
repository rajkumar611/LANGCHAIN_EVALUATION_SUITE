"""Use Case 4: Conversation Memory with Per-Session History."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import MemoryRequest
from ..shared.session_store import LC_MODEL, LC_SESSIONS

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/memory")
def lc_memory(req: MemoryRequest):
    """Demonstrate per-session conversation memory via MessagesPlaceholder.

    Each session_id maintains its own history list. Messages are stored as
    LangChain HumanMessage/AIMessage objects and injected into each prompt.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        if req.session_id not in LC_SESSIONS:
            LC_SESSIONS[req.session_id] = []
        history = LC_SESSIONS[req.session_id]

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a friendly assistant. Remember everything the user tells you."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        answer = (prompt | chat_model | StrOutputParser()).invoke(
            {"history": history, "input": req.message}
        )
        history.append(HumanMessage(content=req.message))
        history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "history": [
                {"role": "user" if isinstance(m, HumanMessage) else "bot", "text": m.content}
                for m in history
            ],
        }
    except Exception as e:
        logger.error("lc_memory error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.delete("/langchain/memory/{session_id}")
def lc_memory_clear(session_id: str):
    """Clear all conversation history for a given session."""
    LC_SESSIONS.pop(session_id, None)
    return {"cleared": True}
