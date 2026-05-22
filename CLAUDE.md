# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**LangChain Evaluation Suite** — a FastAPI web app demonstrating 10 LangChain orchestration patterns and LangGraph workflows. A single `frontend/index.html` SPA calls backend endpoints. No database, no auth.

## Running the app

```bash
pip install -r requirements.txt
python main.py                    # serves on http://localhost:8081
```

`main.py` loads `.env`, validates `ANTHROPIC_API_KEY` is set (exits with a clear message if not), then starts uvicorn. `reload` is enabled only when `ENV=development` (the default).

Optional: add `LANGCHAIN_API_KEY` to `.env` to enable LangSmith tracing automatically.

### Docker

```bash
docker compose up          # reads .env, sets ENV=production, healthchecks /health
```

## Architecture

```
app.py                                  ← FastAPI app; mounts routers; /health; logging setup
main.py                                 ← uvicorn entrypoint; startup validation
src/
  config.py                             ← Pydantic BaseSettings — single source of truth for all env vars
  langchain_orchestration/
    routes.py                           ← 10 LangChain concept demos
frontend/
  index.html                            ← entire SPA (single file)
Dockerfile                              ← multi-stage build
docker-compose.yml                      ← local dev; healthcheck wired to /health
pyproject.toml                          ← ruff, mypy config
.github/workflows/ci.yml                ← runs lint on every push/PR
```

## Configuration (`src/config.py`)

All settings are defined as a `pydantic_settings.BaseSettings` subclass and imported as `from src.config import settings`. Do **not** use `os.getenv()` directly in route files — add new settings to `config.py` instead.

| Setting | Default | Description |
|---|---|---|
| `anthropic_api_key` | *(required)* | Anthropic API key |
| `haiku_model` | `claude-haiku-4-5-20251001` | Model used by LangChain module |
| `port` | `8081` | Server port |
| `env` | `development` | Set to `production` to disable reload |

## LangChain module (`src/langchain_orchestration/routes.py`)

Uses **`langchain-anthropic`** (`ChatAnthropic`, model `claude-haiku-4-5-20251001`).

All LangChain imports are deferred (inside each route function) — keeps startup fast.

| Endpoint | Concept |
|---|---|
| `POST /langchain/prompt` | Prompt templates + `StrOutputParser` |
| `POST /langchain/chaining` | Sequential 3-step chain (translate → summarise → JSON) |
| `POST /langchain/rag` | RAG with a fixed 7-doc FAISS vectorstore (lazy-loaded, in-memory) |
| `POST /langchain/memory` | Per-session conversation history via `MessagesPlaceholder` |
| `POST /langchain/tools` | `bind_tools` with calculator / weather / word-count |
| `POST /langchain/documents` | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` |
| `POST /langchain/parsers` | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` |
| `POST /langchain/agent` | ReAct agent via `langgraph.prebuilt.create_react_agent` |
| `POST /langchain/multiagent` | Two sequential LLM calls (researcher → blog writer) |
| `POST /langchain/langgraph` | `StateGraph` with manager → research → writer → reviewer + conditional edge (max 2 revisions) |

**Request models**: `PromptRequest`, `TextRequest`, `QuestionRequest`, `TopicRequest`, `MemoryRequest`.

`LC_SESSIONS` (dict) stores per-session chat history — cleared with `DELETE /langchain/memory/{session_id}`.

**Calculator tool**: Uses a whitelist-only AST walker (`_safe_eval_math`) — **no `eval()` anywhere**. Only permits `+`, `-`, `*`, `/`, `**`, and numeric literals. Input is pre-validated by regex before parsing.

## Health endpoint

`GET /health` — returns uptime, `len(LC_SESSIONS)`, and model names. No external calls. Used as the Docker healthcheck.

## Key constraints

- **All state is in-memory.** Restarting the server clears chat sessions.
- The LangChain fixed vectorstore (`_lc_vectorstore`) contains only 7 hardcoded AI/ML facts.
- `frontend/index.html` is a single-file SPA — all JS, CSS, and HTML in one file. Edit it directly.
- `app.py` enables CORS with wildcard (`allow_origins=["*"]`) — fine for local dev, not for production.
- LangSmith tracing activates automatically when `LANGCHAIN_API_KEY` is set in `.env`.
- All Pydantic request models enforce `min_length` / `max_length` constraints — do not remove these.
- Every route endpoint wraps its body in `try/except` and returns `JSONResponse(status_code=500)` on failure — maintain this pattern when adding new endpoints.

## Adding a new LangChain endpoint

1. Add a route function in `src/langchain_orchestration/routes.py` with a docstring explaining the pattern.
2. Use an appropriate request model (or create a new one if needed).
3. Wrap the body in `try/except` → `JSONResponse(500)`.
4. Keep imports deferred inside the route function for fast startup.
5. Use `ChatAnthropic` with `settings.haiku_model`.
