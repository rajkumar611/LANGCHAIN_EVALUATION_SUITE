# LangChain Evaluation Suite

A FastAPI web app demonstrating **10 LangChain orchestration patterns** and LangGraph workflows — built to production-grade standards with comprehensive examples of prompt templates, chaining, agents, multiagent systems, and state management.

![CI](https://github.com/rajkumar611/LANGCHAIN_EVALUATION_SUITE/actions/workflows/ci.yml/badge.svg)

---

## Why This Project

This suite showcases production-ready LangChain patterns with:
- **Prompt templates & chaining** — from simple templates to multi-step sequences
- **Retrieval-augmented generation** — LangChain's built-in RAG pattern with fixed vectorstore
- **Conversation memory** — per-session chat history with multiple strategies
- **Tool calling & agents** — ReAct agents with bound tools (calculator, weather, word-count)
- **Output parsing** — structured extraction with JSON, comma-separated lists, and more
- **Multiagent systems** — sequential agent collaboration and handoffs
- **LangGraph workflows** — state machines with conditional edges and human-in-the-loop patterns
- **Production rigour** — typed config, structured logging, input validation, Docker support

---

## 10 LangChain Demos

| Endpoint | Concept | Key Feature |
|---|---|---|
| `POST /langchain/prompt` | Prompt templates + `StrOutputParser` | Template variables, prompt composition |
| `POST /langchain/chaining` | Sequential 3-step chain (translate → summarise → JSON) | Chain composition, multiple LLM calls |
| `POST /langchain/rag` | RAG with a fixed 7-doc FAISS vectorstore | Dense retrieval, context passing to LLM |
| `POST /langchain/memory` | Per-session conversation history via `MessagesPlaceholder` | Stateful chat, session management |
| `POST /langchain/tools` | `bind_tools` with calculator / weather / word-count | Tool calling, structured outputs |
| `POST /langchain/documents` | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` | Text chunking strategies |
| `POST /langchain/parsers` | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` | Output parsing and structuring |
| `POST /langchain/multiagent` | Sequential chain pipeline (researcher → blog writer) | Pipeline composition, multi-step chains |
| `POST /langchain/agent` | ReAct agent via `langgraph.prebuilt.create_react_agent` | Agentic reasoning, iterative tool use |
| `POST /langchain/workflow` | `StateGraph` with manager → research → writer → reviewer | Graph-based workflows, conditional edges |

Each endpoint is self-contained, testable, and demonstrates a specific LangChain capability.

---

## Quick Start

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. Run
python main.py
# → http://localhost:8081
```

### Docker

```bash
# Build and run
docker compose up

# Or without compose
docker build -t langchain-eval .
docker run -p 8081:8080 -e ANTHROPIC_API_KEY=sk-... langchain-eval
```

---

## Project Structure

```
src/langchain_orchestration/
├── __init__.py                         # Router aggregation (imports all 10 endpoints)
├── shared/                             # Cross-cutting concerns (SRP principle)
│   ├── models.py                       # 5 Pydantic request models with validation
│   ├── tools.py                        # Safe math evaluator + tool factories
│   ├── vectorstore.py                  # Lazy FAISS singleton
│   └── session_store.py                # Session dict + model constant
└── endpoints/                          # 10 LangChain use cases (1 file per strategy)
    ├── 01_prompt.py                    # Use Case 1:  Prompt Templates
    ├── 02_chaining.py                  # Use Case 2:  LLM Chaining
    ├── 03_rag.py                       # Use Case 3:  RAG
    ├── 04_memory.py                    # Use Case 4:  Memory & Sessions
    ├── 05_tools.py                     # Use Case 5:  Tools & Function Calling
    ├── 06_documents.py                 # Use Case 6:  Document Processing
    ├── 07_parsers.py                   # Use Case 7:  Output Parsers
    ├── 08_chain_pipeline.py            # Use Case 8:  Sequential Chain Pipeline
    ├── 09_agent.py                     # Use Case 9:  ReAct Agent
    └── 10_workflow.py                  # Use Case 10: LangGraph Workflow

tests/
├── conftest.py                         # TestClient fixture + environment setup
├── test_01_prompt.py                   # 5 tests (validation + integration)
├── test_02_chaining.py                 # 3 tests
├── test_03_rag.py                      # 2 tests
├── test_04_memory.py                   # 6 tests (validation + session management)
├── test_05_tools.py                    # 3 tests
├── test_06_documents.py                # 5 tests (text splitting logic)
├── test_07_parsers.py                  # 3 tests
├── test_08_chain_pipeline.py           # 3 tests
├── test_09_agent.py                    # 3 tests
├── test_10_workflow.py                 # 3 tests
└── test_shared_tools.py                # 10 unit tests (_safe_eval_math)
                                        # Total: 44 tests ✓ ALL PASSING
```

**1-to-1 Mapping**: Each endpoint file (`endpoints/X.py`) has one corresponding test file (`tests/test_X.py`). See [ARCHITECTURE.md](ARCHITECTURE.md) for design principles.

---

## Testing

**44 tests total** — validation, integration, unit, and domain tests organized by use case.

```bash
# Run all tests
pytest tests/ -v

# Run tests for one endpoint
pytest tests/test_prompt.py -v

# Run one test
pytest tests/test_prompt.py::test_prompt_endpoint_exists -v

# Show test coverage
pytest tests/ --cov=src --cov-report=html
```

**Test types**:
- **Validation** (17 tests) — request model constraints (empty, oversized, invalid format)
- **Integration** (10 tests) — endpoint callable (HTTP 200 or 500, not 404)
- **Unit** (10 tests) — pure function tests (`_safe_eval_math` in `test_shared_tools.py`)
- **Domain** (7 tests) — business logic without LLM calls (text splitting, session management)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend SPA |
| `GET` | `/health` | Liveness probe — uptime, active sessions, model name |
| `POST` | `/langchain/prompt` | Prompt templates demo |
| `POST` | `/langchain/chaining` | Chaining demo |
| `POST` | `/langchain/rag` | RAG demo |
| `POST` | `/langchain/memory` | Memory & conversation history demo |
| `POST` | `/langchain/tools` | Tool calling demo |
| `POST` | `/langchain/documents` | Text splitting strategies |
| `POST` | `/langchain/parsers` | Output parsing demo |
| `POST` | `/langchain/multiagent` | Sequential chain pipeline demo |
| `POST` | `/langchain/agent` | ReAct agent demo |
| `POST` | `/langchain/workflow` | LangGraph workflow demo |
| `DELETE` | `/langchain/memory/{session_id}` | Clear session history |

Full interactive docs: `http://localhost:8081/docs`

---

## Key Design Decisions

- **Deferred LangChain imports** — imports inside each route function keep startup fast
- **All state in-memory** — intentional for simplicity; chat sessions reset on server restart
- **Typed config** — `src/config.py` (pydantic-settings) validates all env vars at startup
- **No database, no auth** — focus is on LangChain patterns, not infrastructure
- **LangSmith integration** — automatic tracing when `LANGCHAIN_API_KEY` is set

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (`claude-haiku-4-5-20251001`) |
| Orchestration | LangChain + LangGraph |
| Config | Pydantic Settings |
| Observability | Structured logging |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `HAIKU_MODEL` | `claude-haiku-4-5-20251001` | LLM model |
| `PORT` | `8081` | Server port |
| `LANGCHAIN_API_KEY` | *(optional)* | LangSmith API key for tracing |

---

## Author

**Raj Kumar** — [rajkumar.novsix@gmail.com](mailto:rajkumar.novsix@gmail.com)
