"""LangChain orchestration module with 10 use case endpoints."""

import importlib
from fastapi import APIRouter

from .shared.session_store import LC_SESSIONS

router = APIRouter()

# Load modules with numeric prefixes dynamically
module_names = [
    "01_prompt",
    "02_chaining",
    "03_rag",
    "04_memory",
    "05_tools",
    "06_documents",
    "07_parsers",
    "08_agent",
    "09_multiagent",
    "10_langgraph",
]

for module_name in module_names:
    module = importlib.import_module(f".endpoints.{module_name}", package="src.langchain_orchestration")
    router.include_router(module.router)

__all__ = ["router", "LC_SESSIONS"]
