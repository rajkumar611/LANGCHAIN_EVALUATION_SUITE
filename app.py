"""
LangChain Evaluation Suite — FastAPI app
Run via: python main.py
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.langchain_orchestration import LC_SESSIONS, router as lc_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Enable LangSmith tracing when API key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-evaluation-suite")

_START_TIME = time.time()

app = FastAPI(title="LangChain Evaluation Suite")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(lc_router)
app.mount("/pages", StaticFiles(directory="frontend/pages"), name="pages")


@app.get("/")
def root() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/health")
def health() -> dict:
    """Liveness probe — returns runtime state without calling any external service."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "active_sessions": len(LC_SESSIONS),
        "models": {
            "langchain": settings.haiku_model,
        },
    }


@app.get("/api/patterns/{pattern_name}/explanation")
def get_pattern_explanation(pattern_name: str) -> dict:
    """Fetch the explanation markdown for a specific LangChain pattern."""
    # Validate pattern name to prevent directory traversal
    valid_patterns = {
        "prompt": "01_prompt.md",
        "chaining": "02_chaining.md",
        "rag": "03_rag.md",
        "memory": "04_memory.md",
        "tools": "05_tools.md",
        "documents": "06_documents.md",
        "parsers": "07_parsers.md",
        "agent": "08_agent.md",
        "multiagent": "09_multiagent.md",
        "langgraph": "10_langgraph.md",
    }

    if pattern_name not in valid_patterns:
        raise HTTPException(status_code=404, detail="Pattern not found")

    # Use absolute path relative to this file, not the current working directory
    file_path = Path(__file__).parent / "docs" / "patterns" / valid_patterns[pattern_name]

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Explanation file not found")

    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "pattern": pattern_name,
            "content": content,
        }
    except Exception as e:
        logger.error(f"Error reading explanation file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read explanation")
