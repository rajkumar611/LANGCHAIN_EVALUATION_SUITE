# RAG Evaluation Suite

An interactive FastAPI application demonstrating **5 RAG retrieval strategies**, **10 LangChain orchestration patterns**, and **LLM-based RAG evaluation** — built to production-grade standards.

![CI](https://github.com/your-username/RAG_EVALUATION_SUITE/actions/workflows/ci.yml/badge.svg)

---

## Why This Project

Most RAG demos are toy examples. This project shows:
- **Retrieval depth** — five strategies from naive to graph-based, each with visible pipeline steps
- **Orchestration breadth** — ten LangChain patterns from prompt templates to LangGraph stateful workflows
- **Evaluation** — LLM-as-Judge scoring across Faithfulness, Answer Relevancy, and Context Utilization
- **Observability** — LangSmith tracing integration (optional, zero-config when API key is set)
- **Production rigour** — typed config, structured logging, input validation, safe tool sandboxing, 35 tests, CI pipeline, Docker support

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     index.html (SPA)                    │
│          RAG Strategies │ Evaluation │ LangChain         │
└────────────────┬──────────────┬───────────────┬─────────┘
                 │              │               │
         ┌───────▼──────┐  ┌───▼────┐  ┌───────▼──────────┐
         │  rag/routes  │  │  eval  │  │ langchain_        │
         │              │  │ (same) │  │ orchestration/    │
         │ • naive      │  └────────┘  │ routes            │
         │ • advanced   │              │                   │
         │ • agentic    │              │ • prompt templates │
         │ • hybrid     │              │ • chaining        │
         │ • graph      │              │ • RAG             │
         │ • evaluate   │              │ • memory          │
         └──────┬───────┘              │ • tools           │
                │                      │ • documents       │
         ┌──────▼───────┐              │ • parsers         │
         │ Anthropic SDK│              │ • agent (ReAct)   │
         │ claude-sonnet│              │ • multi-agent     │
         │ (direct)     │              │ • LangGraph       │
         └──────────────┘              └────────┬──────────┘
                                                │
                                       ┌────────▼────────┐
                                       │ langchain-       │
                                       │ anthropic        │
                                       │ claude-haiku     │
                                       └─────────────────┘
```

---

## RAG Strategies

| Strategy | When to Use | Key Technique |
|---|---|---|
| **Naive RAG** | Baseline / simple Q&A | Embed → cosine search → generate |
| **Advanced RAG** | Production Q&A | Query rewrite + hybrid search + LLM re-rank |
| **Agentic RAG** | Multi-hop questions | Tool-calling agent with iterative search |
| **Hybrid RAG** | Keyword + semantic needs | Dense (FAISS) + Sparse (BM25) fused via RRF |
| **Graph RAG** | Relational / connected data | Seed retrieval + BFS graph expansion |

---

## LangChain Orchestration Patterns

| Pattern | Production Use Case |
|---|---|
| Prompt Templates | Standardised, reusable prompt management |
| Chaining | Multi-step pipelines (translate → summarise → format) |
| RAG | Document Q&A with FAISS retriever |
| Memory | Session-aware conversation with `MessagesPlaceholder` |
| Tools | External API integration via `bind_tools` |
| Document Splitters | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` |
| Output Parsers | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` |
| Agent (ReAct) | Autonomous reasoning with `create_react_agent` |
| Multi-Agent | Role-based sequential agents (researcher → writer) |
| LangGraph | Stateful workflows with conditional edges and revision loops |

---

## RAG Evaluation (LLM-as-Judge)

Four RAGAS-aligned metrics evaluated by Claude:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in context? (hallucination detection) |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Utilization** | Did the retrieved chunks contain the right information? |
| **Correctness** *(optional)* | Accuracy vs ground truth answer |

---

## Quick Start

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. (Optional) Enable LangSmith tracing
# Add LANGCHAIN_API_KEY to .env — tracing activates automatically

# 4. Run
python main.py
# → http://localhost:8000
```

### Docker

```bash
# Build and run (reads ANTHROPIC_API_KEY from .env automatically)
docker compose up

# Or without compose
docker build -t rag-eval .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... rag-eval
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run fully offline — all LLM calls are mocked. No API key required.

```
tests/test_rag_utils.py      # Unit tests: chunk_text, vector_search, bm25_search, reciprocal_rank_fusion
tests/test_rag_endpoints.py  # Integration tests: /upload, /rag/naive, /rag/hybrid, /rag/evaluate
```

---

## Project Structure

```
.
├── app.py                              # FastAPI app — routers, logging, /health endpoint
├── main.py                             # Uvicorn entrypoint with startup validation
├── requirements.txt                    # Pinned dependencies
├── pyproject.toml                      # Pytest, ruff (lint + format), mypy config
├── .env.example
├── Dockerfile                          # Multi-stage build; pre-downloads embedding model
├── docker-compose.yml                  # Local dev with healthcheck wired to /health
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # Runs tests + lint on every push / PR
│
├── src/
│   ├── config.py                       # Pydantic BaseSettings — single source of truth for all env vars
│   ├── rag/
│   │   └── routes.py                  # 5 RAG strategies + /upload + /rag/evaluate
│   └── langchain_orchestration/
│       └── routes.py                  # 10 LangChain orchestration patterns
│
├── frontend/
│   └── index.html                     # Single-file SPA
│
└── tests/
    ├── conftest.py                     # Shared fixtures (client, sample_text, uploaded_client)
    ├── test_rag_utils.py               # 16 utility unit tests
    └── test_rag_endpoints.py           # 19 endpoint integration tests
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend SPA |
| `GET` | `/health` | Liveness probe — uptime, doc count, model names |
| `POST` | `/upload` | Upload PDF or TXT; rebuilds all indexes |
| `POST` | `/rag/naive` | Baseline RAG |
| `POST` | `/rag/advanced` | Advanced RAG with re-ranking |
| `POST` | `/rag/agentic` | Agentic tool-calling RAG |
| `POST` | `/rag/hybrid` | Hybrid dense + sparse RAG |
| `POST` | `/rag/graph` | Graph-expanded RAG |
| `POST` | `/rag/evaluate` | LLM-as-Judge evaluation |
| `POST` | `/langchain/*` | 10 LangChain pattern demos |
| `DELETE` | `/langchain/memory/{session_id}` | Clear conversation session |

Full interactive docs: `http://localhost:8000/docs`

---

## Key Design Decisions

- **Two Claude models intentionally**: RAG uses `claude-sonnet-4-6` (higher reasoning for retrieval tasks); LangChain demos use `claude-haiku-4-5-20251001` (faster, cheaper for concept demos)
- **LLM-as-Judge over RAGAS library**: Avoids heavy dependencies; the same evaluation principle, implemented directly with the Anthropic SDK
- **LangSmith zero-config**: Set `LANGCHAIN_API_KEY` in `.env` — all LangChain calls trace automatically, no code changes required
- **All state in-memory**: Intentional for simplicity; upload a PDF before using RAG endpoints (resets on server restart)
- **Safe math sandboxing**: The calculator tool uses a whitelist-only AST walker — no `eval()` anywhere in the codebase
- **Typed config**: `src/config.py` (`pydantic-settings`) validates all env vars at startup; the app exits immediately with a clear message if `ANTHROPIC_API_KEY` is missing

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (Sonnet 4.6 + Haiku 4.5) |
| Orchestration | LangChain + LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Search | FAISS |
| Sparse Search | TF-IDF / BM25 (scikit-learn) |
| Graph | NetworkX |
| Config | Pydantic Settings |
| Observability | LangSmith (optional) + structured logging |
| Testing | pytest + FastAPI TestClient |
| Containerisation | Docker (multi-stage) + Docker Compose |
| CI | GitHub Actions |

---

## Author

**Raj Kumar** — [rajkumar.novsix@gmail.com](mailto:rajkumar.novsix@gmail.com)
