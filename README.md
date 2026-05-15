# RAG Evaluation Suite

An interactive FastAPI application demonstrating **5 RAG retrieval strategies** and **LLM-based RAG evaluation** — built to production-grade standards with a multi-page guided frontend, step-by-step pipeline walkthroughs, and purpose-built sample documents for each strategy.

![CI](https://github.com/your-username/RAG_EVALUATION_SUITE/actions/workflows/ci.yml/badge.svg)

---

## Why This Project

Most RAG demos are toy examples. This project shows:
- **Retrieval depth** — five strategies from naive to graph-based, each with visible pipeline steps
- **Pass / Fail sample documents** — purpose-built documents that demonstrate exactly where each strategy succeeds and where it breaks down
- **Evaluation** — real [RAGAS](https://github.com/explodinggradients/ragas) library scoring across Faithfulness, Answer Relevancy, Context Precision, and optional Correctness
- **Production rigour** — typed config, structured logging, input validation, 69 tests, CI pipeline, Docker support

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              Browser (SPA)                   │
│   index.html  +  frontend/pages/             │
│                                              │
│  Home │ Naive │ Advanced │ Agentic │ Hybrid  │
│                Graph │ Evaluation            │
└─────────────────────┬────────────────────────┘
                      │ HTTP
            ┌─────────▼─────────┐
            │   rag/routes.py   │
            │                   │
            │  POST /upload     │
            │  POST /rag/naive  │
            │  POST /rag/advanced│
            │  POST /rag/agentic│
            │  POST /rag/hybrid │
            │  POST /rag/graph  │
            │  POST /rag/evaluate│
            └─────────┬─────────┘
                      │
            ┌─────────▼─────────┐
            │   Anthropic SDK   │
            │  claude-sonnet-4-6│
            └───────────────────┘
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

Each strategy page explains the pipeline visually, shows step-by-step execution results, and ships with dedicated sample documents to demonstrate pass and fail behaviour.

---

## RAG Evaluation (RAGAS)

Powered by the real [ragas](https://github.com/explodinggradients/ragas) library (v0.4.x) with Claude as the judge LLM. Four metrics across two modes:

| Metric | When scored | What It Measures |
|---|---|---|
| **Faithfulness** | Always | Is the answer grounded in the retrieved context? (hallucination detection) |
| **Answer Relevancy** | Always | Does the answer address the question that was asked? |
| **Context Precision** | With ground truth | Did the retrieved chunks actually contain the right information? |
| **Answer Correctness** | With ground truth | Accuracy against a known reference answer (F1-style claim comparison) |

RAGAS uses Claude via `llm_factory(provider="anthropic")` and the same `all-MiniLM-L6-v2` embeddings used throughout the app. Resources are lazily initialised on the first evaluation call.

---

## Sample Documents

Purpose-built pass / fail document pairs for each strategy. Each set includes a `_prompt.txt` explaining exactly what behaviour to expect and why.

| Strategy | Pass Document | Fail Document | Notes |
|---|---|---|---|
| Naive RAG | `naive_rag_pass.txt` | `naive_rag_fail.txt` | Fail: answer split across distant sections |
| Advanced RAG | `advanced_rag_pass.txt` | `advanced_rag_fail.txt` | Fail: causal chain spread across 5+ sections |
| Agentic RAG | `agentic_rag_pass.txt` | `agentic_rag_fail.txt` | Fail: two similar events create conflicting context |
| Hybrid RAG | `hybrid_rag_pass.txt` | `hybrid_rag_fail.txt` | Pass: exact error code + conceptual query (BM25 + vector both needed) |
| Graph RAG | `graph_rag_pass.txt` | `graph_rag_fail.txt` | Also includes 4-document multi-hop scenario |

General purpose documents (`doc1_ai_research_report.txt`, `doc2_product_operations_manual.txt`) work across all strategies.

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
# → http://localhost:8080
```

### Docker

```bash
# Build and run (reads ANTHROPIC_API_KEY from .env automatically)
docker compose up

# Or without compose
docker build -t rag-eval .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-... rag-eval
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run fully offline — all LLM calls are mocked. No API key required.

```
tests/test_rag_utils.py      # 25 unit tests: chunk_text, vector_search, bm25_search, reciprocal_rank_fusion, ctx_prompt
tests/test_rag_endpoints.py  # 44 integration tests: /upload, /rag/naive, /rag/advanced, /rag/agentic,
                             #                       /rag/hybrid, /rag/graph, /rag/evaluate (RAGAS mocked)
```

---

## Project Structure

```
.
├── app.py                              # FastAPI app — routers, logging, /health endpoint
├── main.py                             # Uvicorn entrypoint with startup validation
├── requirements.txt
├── pyproject.toml                      # Pytest, ruff (lint + format), mypy config
├── Dockerfile                          # Multi-stage build; pre-downloads embedding model
├── docker-compose.yml
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # Runs tests + lint on every push / PR
│
├── src/
│   ├── config.py                       # Pydantic BaseSettings — single source of truth for all env vars
│   └── rag/
│       └── routes.py                   # 5 RAG strategies + /upload + /rag/evaluate
│
├── frontend/
│   ├── index.html                      # SPA shell — navigation and page routing
│   └── pages/
│       ├── home.html                   # Overview: what RAG is, how strategies differ
│       ├── naive-rag.html
│       ├── advanced-rag.html
│       ├── agentic-rag.html
│       ├── hybrid-rag.html
│       ├── graph-rag.html
│       ├── rag-evaluation.html
│       └── summary.html
│
├── sample_docs/
│   ├── naive_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── advanced_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── hybrid_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── agentic_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── graph_rag_pass.txt / _fail.txt / _prompt.txt
│   ├── graph_rag_doc1_teams.txt        # Multi-document Graph RAG scenario
│   ├── graph_rag_doc2_tech_stack.txt
│   ├── graph_rag_doc3_projects.txt
│   ├── graph_rag_doc4_security.txt
│   ├── doc1_ai_research_report.txt     # General purpose documents
│   └── doc2_product_operations_manual.txt
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
| `GET` | `/health` | Liveness probe — uptime, doc count, model name |
| `POST` | `/upload` | Upload PDF or TXT; rebuilds all indexes |
| `POST` | `/rag/naive` | Baseline RAG |
| `POST` | `/rag/advanced` | Advanced RAG with query rewrite and re-ranking |
| `POST` | `/rag/agentic` | Agentic tool-calling RAG |
| `POST` | `/rag/hybrid` | Hybrid dense + sparse RAG |
| `POST` | `/rag/graph` | Graph-expanded RAG |
| `POST` | `/rag/evaluate` | LLM-as-Judge evaluation |

Full interactive docs: `http://localhost:8080/docs`

---

## Key Design Decisions

- **Real RAGAS evaluation** — uses the `ragas==0.4.x` library with Claude as the judge LLM; resources are lazily initialised on the first `/rag/evaluate` call so startup is fast
- **All state in-memory** — intentional for simplicity; upload a document before querying (state resets on server restart)
- **Typed config** — `src/config.py` (pydantic-settings) validates all env vars at startup; the app exits immediately with a clear message if `ANTHROPIC_API_KEY` is missing
- **Pass / Fail sample documents** — each strategy ships with documents designed to expose the strategy's specific failure mode, not just documents that work

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Search | NumPy dot product (FAISS used in LangChain demo only) |
| Sparse Search | TF-IDF / BM25 (scikit-learn) |
| Evaluation | RAGAS (`ragas==0.4.x`) |
| Graph | NetworkX |
| Config | Pydantic Settings |
| Observability | Structured logging |
| Testing | pytest + FastAPI TestClient |
| Containerisation | Docker (multi-stage) + Docker Compose |
| CI | GitHub Actions |

---

## Future Enhancements

### Citations
Currently, retrieved chunks are displayed alongside every answer, but the chunks themselves carry no information about which file they came from or where in that file they appeared. In a production system serving end users who only see the final answer, this is a gap — there is no way to trace a claim back to its source document.

The planned change is: at upload time, tag each chunk with its source filename and position. Pass that metadata through retrieval so every chunk in the response carries a `source` field. Update the LLM prompt to instruct inline citation markers (`[1]`, `[2]`), and show the filename on each chunk card in the UI.

This becomes essential when multiple documents are uploaded simultaneously or when the app is used in domains where every claim must be verifiable.

---

## Author

**Raj Kumar** — [rajkumar.novsix@gmail.com](mailto:rajkumar.novsix@gmail.com)
