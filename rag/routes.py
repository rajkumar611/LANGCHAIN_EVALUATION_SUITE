import io, json, re
import numpy as np
import networkx as nx
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos

router = APIRouter()

# ── Global state ──────────────────────────────────────────────────────────────
DOCS: list[str] = []
DOC_EMBS: np.ndarray = np.array([])
tfidf = TfidfVectorizer(stop_words="english")
TFIDF_MAT = None
G = nx.Graph()

print("Loading embedding model (first run downloads ~80 MB)…")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = anthropic.Anthropic()
CLAUDE   = "claude-sonnet-4-6"

class Q(BaseModel):
    query: str

def rebuild_indexes(docs: list[str]):
    global DOCS, DOC_EMBS, TFIDF_MAT
    DOCS      = docs
    DOC_EMBS  = embedder.encode(DOCS, normalize_embeddings=True)
    TFIDF_MAT = tfidf.fit_transform(DOCS)
    G.clear()
    for i in range(len(DOCS)):
        G.add_node(f"d{i}", text=DOCS[i])
    for i in range(len(DOCS) - 1):
        G.add_edge(f"d{i}", f"d{i+1}")

def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    text   = re.sub(r'\n{3,}', '\n\n', text.strip())
    paras  = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 40]
    chunks = []
    for para in paras:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sents, cur = re.split(r'(?<=[.!?])\s+', para), ""
            for s in sents:
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = (cur + " " + s).strip() if cur else s
                else:
                    if cur: chunks.append(cur)
                    cur = s
            if cur: chunks.append(cur)
    return chunks[:300]

def vsearch(query: str, k: int = 5) -> list[dict]:
    qe     = embedder.encode([query], normalize_embeddings=True)
    scores = (qe @ DOC_EMBS.T)[0]
    top    = np.argsort(scores)[::-1][:k]
    return [{"id": int(i), "text": DOCS[i], "score": round(float(scores[i]), 4)} for i in top]

def bsearch(query: str, k: int = 5) -> list[dict]:
    qv     = tfidf.transform([query])
    scores = sk_cos(qv, TFIDF_MAT)[0]
    top    = np.argsort(scores)[::-1][:k]
    return [{"id": int(i), "text": DOCS[i], "score": round(float(scores[i]), 4)} for i in top]

def rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[int, float] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            scores[item["id"]] = scores.get(item["id"], 0) + 1 / (rank + k)
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{"id": i, "text": DOCS[i], "score": round(scores[i], 6)} for i in order]

def llm(prompt: str, max_tokens: int = 512) -> str:
    r = client.messages.create(
        model=CLAUDE, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()

def no_docs_response():
    return {"answer": "No document uploaded yet. Please upload a PDF or TXT file first.",
            "docs": [], "steps": []}

def ctx_prompt(docs: list[dict], question: str) -> str:
    ctx = "\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer in 2-3 sentences:"


# ── Upload ────────────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    fname   = file.filename or ""

    if fname.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return {"error": f"PDF error: {e}"}
    else:
        text = content.decode("utf-8", errors="ignore")

    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No readable text found in file."}

    rebuild_indexes(chunks)
    return {"filename": fname, "chunks": len(chunks)}


# ════════════════════════════════════════════════════════════════════════════════
# 1 · NAIVE RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/naive")
def naive_rag(q: Q):
    if not DOCS: return no_docs_response()
    steps = [{"step": "1. Embed Query",     "detail": f'Encode "{q.query}" → 384-dim vector'}]
    docs  = vsearch(q.query, k=3)
    steps.append({"step": "2. Vector Search",  "detail": f"Cosine similarity over {len(DOCS)} docs → top 3"})
    steps.append({"step": "3. Augment Prompt", "detail": "Prepend top-3 chunks to the user question"})
    ans   = llm(ctx_prompt(docs, q.query))
    steps.append({"step": "4. Generate",       "detail": "LLM reads augmented prompt → answer"})
    return {"answer": ans, "docs": docs, "steps": steps}


# ════════════════════════════════════════════════════════════════════════════════
# 2 · ADVANCED RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/advanced")
def advanced_rag(q: Q):
    if not DOCS: return no_docs_response()
    steps = []

    rewritten = llm(
        f"Rewrite this query to be specific and retrieval-optimized. "
        f"Return ONLY the rewritten query.\n\nOriginal: {q.query}", max_tokens=128
    )
    steps.append({"step": "1. Query Rewriting", "detail": f'"{q.query}" → "{rewritten}"'})

    vr = vsearch(rewritten, k=5)
    br = bsearch(rewritten, k=5)
    steps.append({"step": "2. Hybrid Search",   "detail": "Dense vector + sparse BM25 in parallel"})

    fused = rrf([vr, br])[:5]
    steps.append({"step": "3. RRF Fusion",       "detail": "Reciprocal Rank Fusion: score = Σ 1/(rank+60)"})

    passages = "\n".join(f"ID {d['id']}: {d['text']}" for d in fused)
    raw = llm(
        f"Score each passage for relevance to '{q.query}' (0-10). "
        f"Return JSON array [{{\"id\": N, \"score\": X}}]. JSON only.\n\n{passages}",
        max_tokens=256,
    )
    try:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            id2sc = {r["id"]: r["score"] for r in json.loads(m.group())}
            fused.sort(key=lambda d: id2sc.get(d["id"], 0), reverse=True)
    except Exception:
        pass
    steps.append({"step": "4. LLM Re-ranking",  "detail": "LLM scores each candidate — precision over recall"})

    docs = fused[:3]
    ans  = llm(ctx_prompt(docs, q.query))
    steps.append({"step": "5. Generate",         "detail": "Answer from rewritten query + re-ranked context"})
    return {"answer": ans, "docs": docs, "steps": steps, "rewritten_query": rewritten}


# ════════════════════════════════════════════════════════════════════════════════
# 3 · AGENTIC RAG
# ════════════════════════════════════════════════════════════════════════════════
TOOLS = [{
    "name": "search_knowledge_base",
    "description": "Search the knowledge base. Call multiple times with different queries if needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    },
}]

@router.post("/rag/agentic")
def agentic_rag(q: Q):
    if not DOCS: return no_docs_response()
    steps        = [{"step": "0. Init Agent", "detail": f'Agent receives task: "{q.query}"'}]
    all_docs: list[dict] = []
    messages     = [{"role": "user", "content":
        f"Answer this question by searching the knowledge base. "
        f"Search multiple times with different queries if needed.\n\nQuestion: {q.query}"}]
    final_answer = ""

    for turn in range(5):
        resp      = client.messages.create(model=CLAUDE, max_tokens=512, tools=TOOLS, messages=messages)
        texts     = [b.text for b in resp.content if hasattr(b, "text") and b.text]
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if resp.stop_reason == "end_turn" or not tool_uses:
            final_answer = texts[0] if texts else "No answer generated."
            steps.append({"step": f"{turn+1}. Done", "detail": "Agent satisfied — returning final answer"})
            break

        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        for tu in tool_uses:
            sq      = tu.input.get("query", q.query)
            tk      = min(int(tu.input.get("top_k", 3)), 5)
            results = vsearch(sq, k=tk)
            all_docs.extend(results)
            steps.append({"step": f"{turn+1}. Tool Call",
                          "detail": f'search_knowledge_base(query="{sq}", top_k={tk}) → {len(results)} hits'})
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": json.dumps([{"id": r["id"], "text": r["text"]} for r in results]),
            })
        messages.append({"role": "user", "content": tool_results})
    else:
        final_answer = "Max iterations reached."

    seen: set[int] = set()
    unique = [d for d in all_docs if not (d["id"] in seen or seen.add(d["id"]))]  # type: ignore
    return {"answer": final_answer, "docs": unique[:5], "steps": steps}


# ════════════════════════════════════════════════════════════════════════════════
# 4 · HYBRID RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/hybrid")
def hybrid_rag(q: Q):
    if not DOCS: return no_docs_response()
    steps = []

    vr = vsearch(q.query, k=5)
    steps.append({"step": "1. Dense Retrieval",  "detail": "Semantic vector search → top 5 by cosine similarity"})

    br = bsearch(q.query, k=5)
    steps.append({"step": "2. Sparse Retrieval", "detail": "BM25 keyword match → top 5 by TF-IDF weight"})

    fused = rrf([vr, br])[:3]
    steps.append({"step": "3. RRF Fusion",        "detail": "1/(rank+60) — prevents any single list from dominating"})

    v_ids = {r["id"] for r in vr}
    b_ids = {r["id"] for r in br}
    for d in fused:
        d["sources"] = (["vector"] if d["id"] in v_ids else []) + (["bm25"] if d["id"] in b_ids else [])

    ans = llm(ctx_prompt(fused, q.query))
    steps.append({"step": "4. Generate", "detail": "LLM answers from hybrid-fused top-3 docs"})
    return {"answer": ans, "docs": fused, "steps": steps,
            "vector_results": vr, "bm25_results": br}


# ════════════════════════════════════════════════════════════════════════════════
# 5 · GRAPH RAG
# ════════════════════════════════════════════════════════════════════════════════
@router.post("/rag/graph")
def graph_rag(q: Q):
    if not DOCS: return no_docs_response()
    steps = []

    seeds       = vsearch(q.query, k=2)
    seed_nodes  = {f"d{r['id']}" for r in seeds}
    seed_labels = [f"d{r['id']}" for r in seeds]
    steps.append({"step": "1. Seed Retrieval",
                  "detail": f"Vector search finds seeds: {seed_labels}"})

    visited  = set(seed_nodes)
    frontier = set(seed_nodes)
    for hop in range(2):
        new_f: set[str] = set()
        for node in frontier:
            new_f.update(n for n in G.neighbors(node) if n not in visited)
        visited  |= new_f
        frontier  = new_f
        steps.append({"step": f"2.{hop+1} Graph Hop {hop+1}",
                      "detail": f"BFS adds {len(new_f)} neighbors → {len(visited)} total nodes visited"})

    doc_ids = [int(n[1:]) for n in visited if n.startswith("d")]
    if doc_ids:
        qe   = embedder.encode([q.query], normalize_embeddings=True)
        sc   = (qe @ DOC_EMBS[doc_ids].T)[0]
        best = sorted(zip(doc_ids, sc.tolist()), key=lambda x: x[1], reverse=True)[:3]
        docs = [{"id": i, "text": DOCS[i], "score": round(s, 4)} for i, s in best]
    else:
        docs = seeds[:3]
    steps.append({"step": "3. Re-score",
                  "detail": f"Re-rank {len(doc_ids)} graph-expanded candidates by vector similarity"})

    edges: list[list[str]] = []
    sub = {f"d{d['id']}" for d in docs} | seed_nodes
    for n in sub:
        for nb in G.neighbors(n):
            if nb in visited:
                edges.append([n, nb])

    ans = llm(ctx_prompt(docs, q.query))
    steps.append({"step": "4. Generate", "detail": "Answer enriched via multi-hop graph traversal"})
    return {"answer": ans, "docs": docs, "steps": steps,
            "graph_edges": edges[:20], "visited_nodes": list(visited)}
