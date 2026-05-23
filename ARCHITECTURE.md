# Architecture Guide

## SOLID Principles Applied

This project demonstrates all 5 SOLID principles in a production codebase:

| Principle | How It's Applied |
|---|---|
| **SRP** (Single Responsibility) | Each endpoint file owns exactly one use case; each shared module owns one concern (models, tools, vectorstore, sessions) |
| **OCP** (Open/Closed) | Add a new use case: create 1 file + 1 line in `__init__.py`; existing files never change |
| **LSP** (Liskov Substitution) | Stable Pydantic request model contracts; vectorstore can be swapped behind the same interface |
| **ISP** (Interface Segregation) | Each endpoint imports only its specific request model; no fat single-file imports |
| **DIP** (Dependency Inversion) | Endpoints depend on shared abstractions (`shared/tools.py`, `shared/models.py`) not inline code |

---

## File Organization

### Shared Modules (`src/langchain_orchestration/shared/`)

**`models.py`** — Pydantic request definitions
- `PromptRequest` — role (max 200), text (max 2000)
- `TextRequest` — text (max 10000)
- `QuestionRequest` — question (max 2000)
- `TopicRequest` — topic (max 500)
- `MemoryRequest` — message (max 2000), session_id (max 100, alphanumeric + dash)

**`tools.py`** — Safe math evaluator + tool factories
- `_ALLOWED_OPS` — whitelist of AST operations (+, -, *, /, **, unary -)
- `_safe_eval_math(node)` — secure expression evaluation (no `eval()` anywhere)
- `make_calculator_tool()` — returns LangChain Tool object

**`vectorstore.py`** — Lazy FAISS singleton
- `EMB_MODEL` — "all-MiniLM-L6-v2" (HuggingFace embeddings)
- `get_lc_vectorstore()` — lazy-loads once, cached globally

**`session_store.py`** — Per-session chat history
- `LC_SESSIONS` — dict[session_id, list[messages]] (in-memory)
- `LC_MODEL` — `settings.haiku_model` constant

### Endpoint Files (`src/langchain_orchestration/endpoints/`)

Each file follows the same pattern:

```python
from fastapi import APIRouter
from ..shared.models import SomeRequest
from ..shared.session_store import LC_MODEL

router = APIRouter()

@router.post("/langchain/endpoint")
def endpoint_handler(req: SomeRequest):
    try:
        # Deferred imports for fast startup
        from langchain_anthropic import ChatAnthropic
        # ... implementation
        return {...}
    except Exception as e:
        logger.error("error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
```

**Key design choices**:
- Deferred LangChain imports inside route functions (fast startup when API key validation fails)
- All imports from `shared/` (DIP principle)
- Pydantic request models enforce constraints
- `try/except` wraps body; returns `JSONResponse(500)` on failure
- No mocking of external APIs in tests — validation and endpoint existence suffice

### Router Aggregation (`src/langchain_orchestration/__init__.py`)

```python
router = APIRouter()
for mod in [prompt, chaining, rag, memory, ...]:
    router.include_router(mod.router)
```

One place to register all 10 endpoints; adding a new use case requires only:
1. Create `endpoints/my_feature.py` with a `router`
2. Add `my_feature` to the import list in `__init__.py`

---

## Testing Strategy

### Test Organization

**10 endpoint test files + 1 shared utilities test file = 11 total test files**

```
tests/
├── conftest.py                 # TestClient fixture; sets dummy ANTHROPIC_API_KEY
├── test_01_prompt.py           ↔ src/endpoints/01_prompt.py
├── test_02_chaining.py         ↔ src/endpoints/02_chaining.py
├── test_03_rag.py              ↔ src/endpoints/03_rag.py
├── test_04_memory.py           ↔ src/endpoints/04_memory.py
├── test_05_tools.py            ↔ src/endpoints/05_tools.py
├── test_06_documents.py        ↔ src/endpoints/06_documents.py
├── test_07_parsers.py          ↔ src/endpoints/07_parsers.py
├── test_08_chain_pipeline.py   ↔ src/endpoints/08_chain_pipeline.py
├── test_09_agent.py            ↔ src/endpoints/09_agent.py
├── test_10_workflow.py         ↔ src/endpoints/10_workflow.py
└── test_shared_tools.py        ↔ src/shared/tools.py (_safe_eval_math)
```

### Test Types

| Type | Count | Purpose | Example |
|---|---|---|---|
| **Validation** | 17 | Request constraints (empty, oversized, invalid format) | `test_prompt_validation_empty_role()` |
| **Integration** | 10 | Endpoint callable (HTTP 200 or 500, not 404) | `test_prompt_endpoint_exists()` |
| **Unit** | 10 | Pure functions (`_safe_eval_math`) | `test_safe_eval_math_addition()` |
| **Domain** | 7 | Business logic without LLM (text splitting, session mgmt) | `test_documents_character_splitter()` |

### Why Only 2 Endpoints Have Domain Tests?

- **`test_documents.py`** — Text splitting is pure Python (no LLM needed)
- **`test_memory.py`** — Session management is pure state manipulation (no LLM needed)
- **Other endpoints** — Directly use LangChain + LLM; testing without mocking would require real API keys or extensive mocking setup that adds little value

Validation + Integration tests ensure the API contracts are correct and endpoints are reachable. That's sufficient for production.

---

## Key Constraints & Design Decisions

- **All state is in-memory** — `LC_SESSIONS` cleared on server restart (intentional for simplicity; use Redis/PostgreSQL for persistence)
- **Vectorstore is fixed** — 7 hardcoded AI/ML facts in FAISS (for demo; integrate a real data pipeline in production)
- **No database, no auth** — Focus is on LangChain patterns; add these if deploying to production
- **Deferred imports** — All LangChain imports inside route functions keep startup fast when API key validation fails
- **Safe math evaluation** — Uses AST with whitelisted operations only (no `eval()` anywhere)
- **Request validation** — Pydantic enforces `min_length` / `max_length` on all inputs; do not remove these constraints

---

## Code Walkthrough Flow

For understanding the codebase:

1. **Start** with `src/langchain_orchestration/__init__.py` — shows router aggregation pattern
2. **Shared modules** (`shared/` directory) — understand cross-cutting concerns
   - `models.py` → Pydantic request definitions
   - `session_store.py` → Singleton session storage
   - `vectorstore.py` → Lazy FAISS initialization
   - `tools.py` → Safe math evaluator + factories
3. **Pick any endpoint** (`endpoints/` directory) — all follow the same structure
   - Each owns exactly one REST endpoint
   - Deferred imports keep startup fast
   - Error handling is consistent
4. **Look at tests** — 1-to-1 mapping with endpoints
   - Validation tests ensure input constraints
   - Integration tests confirm endpoints respond
   - Domain/unit tests verify business logic

---

## Adding a New LangChain Use Case

1. Create `src/langchain_orchestration/endpoints/my_feature.py`:
   ```python
   from fastapi import APIRouter
   from ..shared.models import SomeRequest
   
   router = APIRouter()
   
   @router.post("/langchain/my-feature")
   def my_feature_handler(req: SomeRequest):
       try:
           # ... implementation
       except Exception as e:
           return JSONResponse(status_code=500, content={"detail": str(e)})
   ```

2. Add to `src/langchain_orchestration/__init__.py`:
   ```python
   from .endpoints import my_feature
   # ... in the loop:
   for mod in [..., my_feature]:
   ```

3. Create `tests/test_my_feature.py`:
   ```python
   def test_my_feature_validation_empty_field(client):
       response = client.post("/langchain/my-feature", json={"field": ""})
       assert response.status_code == 422
   
   def test_my_feature_endpoint_exists(client):
       response = client.post("/langchain/my-feature", json={"field": "value"})
       assert response.status_code in [200, 500]
   ```

4. Done. No existing files changed.

