"""LangChain orchestration module with 10 use case endpoints."""

from fastapi import APIRouter

from .endpoints import (
    agent,
    chaining,
    documents,
    langgraph_flow,
    memory,
    multiagent,
    parsers,
    prompt,
    rag,
    tool_calling,
)
from .shared.session_store import LC_SESSIONS

router = APIRouter()

for module in [
    prompt,
    chaining,
    rag,
    memory,
    tool_calling,
    documents,
    parsers,
    agent,
    multiagent,
    langgraph_flow,
]:
    router.include_router(module.router)

__all__ = ["router", "LC_SESSIONS"]
