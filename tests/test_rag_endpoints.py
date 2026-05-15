"""
Integration tests for RAG API endpoints.
LLM calls are mocked so tests run without an Anthropic API key or network.
"""

import json
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

MOCK_ANSWER = "This is a mocked LLM answer for testing purposes."


def _mock_llm(prompt: str, max_tokens: int = 512) -> str:
    return MOCK_ANSWER


# ── /upload ───────────────────────────────────────────────────────────────────
class TestUpload:
    def test_upload_txt_success(self, client, sample_text):
        resp = client.post(
            "/upload",
            files=[("files", ("doc.txt", sample_text.encode(), "text/plain"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"][0]["filename"] == "doc.txt"
        assert data["files"][0]["chunks"] > 0
        assert data["total_chunks"] > 0

    def test_upload_multiple_files_success(self, client, sample_text):
        resp = client.post(
            "/upload",
            files=[
                ("files", ("a.txt", sample_text.encode(), "text/plain")),
                ("files", ("b.txt", sample_text.encode(), "text/plain")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["files"]) == 2
        assert data["total_chunks"] == data["files"][0]["chunks"] + data["files"][1]["chunks"]

    def test_upload_empty_file_returns_error(self, client):
        resp = client.post(
            "/upload",
            files=[("files", ("empty.txt", b"", "text/plain"))],
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_upload_whitespace_only_returns_error(self, client):
        resp = client.post(
            "/upload",
            files=[("files", ("blank.txt", b"   \n\n   ", "text/plain"))],
        )
        assert resp.status_code == 200
        assert "error" in resp.json()


# ── /rag/naive ────────────────────────────────────────────────────────────────
class TestNaiveRag:
    def test_no_docs_returns_guidance(self, client):
        # Reset global state to simulate no uploads
        import src.rag.routes as r

        original = r.DOCS[:]
        r.DOCS.clear()
        resp = client.post("/rag/naive", json={"query": "what is RAG?"})
        r.DOCS.extend(original)
        assert resp.status_code == 200
        assert "No document" in resp.json()["answer"]

    def test_returns_expected_keys(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "what is vector search?"})
        assert resp.status_code == 200
        data = resp.json()
        assert {"answer", "docs", "steps"} <= data.keys()

    def test_answer_is_string(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "explain BM25"})
        assert isinstance(resp.json()["answer"], str)

    def test_docs_list_non_empty(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "chunking"})
        assert len(resp.json()["docs"]) > 0

    def test_steps_list_non_empty(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/naive", json={"query": "RAG pipeline"})
        assert len(resp.json()["steps"]) > 0

    def test_query_too_long_rejected(self, client):
        resp = client.post("/rag/naive", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_empty_query_rejected(self, client):
        resp = client.post("/rag/naive", json={"query": ""})
        assert resp.status_code == 422


# ── /rag/advanced ─────────────────────────────────────────────────────────────
class TestAdvancedRag:
    def test_no_docs_returns_guidance(self, client):
        import src.rag.routes as r

        original = r.DOCS[:]
        r.DOCS.clear()
        resp = client.post("/rag/advanced", json={"query": "what is RAG?"})
        r.DOCS.extend(original)
        assert resp.status_code == 200
        assert "No document" in resp.json()["answer"]

    def test_returns_expected_keys(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/advanced", json={"query": "what is vector search?"})
        assert resp.status_code == 200
        data = resp.json()
        assert {"answer", "docs", "steps", "rewritten_query"} <= data.keys()

    def test_rewritten_query_is_string(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/advanced", json={"query": "explain BM25"})
        assert isinstance(resp.json()["rewritten_query"], str)

    def test_steps_include_rewrite_and_fusion(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/advanced", json={"query": "chunking"})
        step_names = [s["step"].lower() for s in resp.json()["steps"]]
        assert any("rewriting" in s or "rewrite" in s for s in step_names)
        assert any("fusion" in s or "rrf" in s for s in step_names)

    def test_query_too_long_rejected(self, client):
        resp = client.post("/rag/advanced", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_empty_query_rejected(self, client):
        resp = client.post("/rag/advanced", json={"query": ""})
        assert resp.status_code == 422


# ── /rag/agentic ──────────────────────────────────────────────────────────────
class TestAgenticRag:
    def test_no_docs_returns_guidance(self, client):
        import src.rag.routes as r

        original = r.DOCS[:]
        r.DOCS.clear()
        resp = client.post("/rag/agentic", json={"query": "what is RAG?"})
        r.DOCS.extend(original)
        assert resp.status_code == 200
        assert "No document" in resp.json()["answer"]

    def test_returns_expected_keys(self, uploaded_client):
        mock_response = _build_agentic_mock_response()
        with patch("src.rag.routes.client.messages.create", return_value=mock_response):
            resp = uploaded_client.post("/rag/agentic", json={"query": "what is vector search?"})
        assert resp.status_code == 200
        data = resp.json()
        assert {"answer", "docs", "steps"} <= data.keys()

    def test_answer_is_string(self, uploaded_client):
        mock_response = _build_agentic_mock_response()
        with patch("src.rag.routes.client.messages.create", return_value=mock_response):
            resp = uploaded_client.post("/rag/agentic", json={"query": "explain BM25"})
        assert isinstance(resp.json()["answer"], str)

    def test_docs_list_is_list(self, uploaded_client):
        mock_response = _build_agentic_mock_response()
        with patch("src.rag.routes.client.messages.create", return_value=mock_response):
            resp = uploaded_client.post("/rag/agentic", json={"query": "chunking"})
        assert isinstance(resp.json()["docs"], list)

    def test_query_too_long_rejected(self, client):
        resp = client.post("/rag/agentic", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_empty_query_rejected(self, client):
        resp = client.post("/rag/agentic", json={"query": ""})
        assert resp.status_code == 422


def _build_agentic_mock_response():
    """Build a minimal mock Anthropic response that triggers end_turn (no tool use)."""
    from unittest.mock import MagicMock

    block = MagicMock()
    block.type = "text"
    block.text = MOCK_ANSWER
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


# ── /rag/graph ────────────────────────────────────────────────────────────────
class TestGraphRag:
    def test_no_docs_returns_guidance(self, client):
        import src.rag.routes as r

        original = r.DOCS[:]
        r.DOCS.clear()
        resp = client.post("/rag/graph", json={"query": "what is RAG?"})
        r.DOCS.extend(original)
        assert resp.status_code == 200
        assert "No document" in resp.json()["answer"]

    def test_returns_expected_keys(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/graph", json={"query": "what is vector search?"})
        assert resp.status_code == 200
        data = resp.json()
        assert {"answer", "docs", "steps", "graph_edges", "visited_nodes"} <= data.keys()

    def test_visited_nodes_non_empty(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/graph", json={"query": "embeddings"})
        assert len(resp.json()["visited_nodes"]) > 0

    def test_steps_include_seed_and_hop(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/graph", json={"query": "RAG pipeline"})
        step_names = [s["step"] for s in resp.json()["steps"]]
        assert any("Seed" in s or "seed" in s for s in step_names)
        assert any("Hop" in s or "hop" in s for s in step_names)

    def test_query_too_long_rejected(self, client):
        resp = client.post("/rag/graph", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_empty_query_rejected(self, client):
        resp = client.post("/rag/graph", json={"query": ""})
        assert resp.status_code == 422


# ── /rag/hybrid ───────────────────────────────────────────────────────────────
class TestHybridRag:
    def test_no_docs_returns_guidance(self, client):
        import src.rag.routes as r

        original = r.DOCS[:]
        r.DOCS.clear()
        resp = client.post("/rag/hybrid", json={"query": "what is RAG?"})
        r.DOCS.extend(original)
        assert resp.status_code == 200
        assert "No document" in resp.json()["answer"]

    def test_returns_expected_keys(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "what is vector search?"})
        assert resp.status_code == 200
        data = resp.json()
        assert {"answer", "docs", "steps", "vector_results", "bm25_results"} <= data.keys()

    def test_returns_both_result_sets(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "retrieval"})
        assert resp.status_code == 200
        data = resp.json()
        assert "vector_results" in data
        assert "bm25_results" in data

    def test_docs_have_source_annotation(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "embeddings"})
        for doc in resp.json()["docs"]:
            assert "sources" in doc
            assert len(doc["sources"]) > 0

    def test_answer_is_string(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "BM25"})
        assert isinstance(resp.json()["answer"], str)

    def test_steps_list_non_empty(self, uploaded_client):
        with patch("src.rag.routes.llm", side_effect=_mock_llm):
            resp = uploaded_client.post("/rag/hybrid", json={"query": "chunking"})
        assert len(resp.json()["steps"]) > 0

    def test_query_too_long_rejected(self, client):
        resp = client.post("/rag/hybrid", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_empty_query_rejected(self, client):
        resp = client.post("/rag/hybrid", json={"query": ""})
        assert resp.status_code == 422


# ── /rag/evaluate ─────────────────────────────────────────────────────────────
def _make_ragas_result(scores: dict) -> MagicMock:
    """Return a mock RAGAS evaluate() result with the given scores dict."""
    result = MagicMock()
    result.scores = [scores]
    return result


def _ragas_patches(scores: dict) -> list:
    """All patches needed to isolate the RAGAS evaluation endpoint from real LLM calls."""
    return [
        patch("src.rag.routes._get_ragas_resources", return_value=(MagicMock(), MagicMock())),
        patch("ragas.evaluate", return_value=_make_ragas_result(scores)),
        patch("ragas.EvaluationDataset", MagicMock()),
        patch("ragas.dataset_schema.SingleTurnSample", MagicMock()),
        patch("ragas.metrics.collections.Faithfulness", MagicMock()),
        patch("ragas.metrics.collections.AnswerRelevancy", MagicMock()),
        patch("ragas.metrics.collections.ContextPrecision", MagicMock()),
        patch("ragas.metrics.collections.AnswerCorrectness", MagicMock()),
    ]


class TestRagEvaluate:
    BASE_PAYLOAD = {
        "question": "What is RAG?",
        "answer": "RAG combines retrieval with generation.",
        "contexts": ["RAG stands for Retrieval Augmented Generation."],
    }

    _BASE_SCORES = {"faithfulness": 0.9, "answer_relevancy": 0.8}
    _FULL_SCORES = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.85,
        "answer_correctness": 0.75,
    }

    def test_returns_scores_dict(self, client):
        with ExitStack() as stack:
            for p in _ragas_patches(self._BASE_SCORES):
                stack.enter_context(p)
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        assert resp.status_code == 200
        assert "scores" in resp.json()

    def test_two_core_metrics_without_ground_truth(self, client):
        with ExitStack() as stack:
            for p in _ragas_patches(self._BASE_SCORES):
                stack.enter_context(p)
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        scores = resp.json()["scores"]
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_utilization" not in scores
        assert "correctness" not in scores

    def test_four_metrics_with_ground_truth(self, client):
        payload = {**self.BASE_PAYLOAD, "ground_truth": "RAG is Retrieval Augmented Generation."}
        with ExitStack() as stack:
            for p in _ragas_patches(self._FULL_SCORES):
                stack.enter_context(p)
            resp = client.post("/rag/evaluate", json=payload)
        scores = resp.json()["scores"]
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_utilization" in scores
        assert "correctness" in scores

    def test_overall_is_average_of_scores(self, client):
        with ExitStack() as stack:
            for p in _ragas_patches(self._BASE_SCORES):
                stack.enter_context(p)
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        data = resp.json()
        scores = list(data["scores"].values())
        expected = round(sum(scores) / len(scores), 3)
        assert abs(data["overall"] - expected) < 0.001

    def test_scores_bounded_zero_to_one(self, client):
        with ExitStack() as stack:
            for p in _ragas_patches(self._BASE_SCORES):
                stack.enter_context(p)
            resp = client.post("/rag/evaluate", json=self.BASE_PAYLOAD)
        for val in resp.json()["scores"].values():
            assert 0.0 <= val <= 1.0

    def test_missing_question_rejected(self, client):
        payload = {k: v for k, v in self.BASE_PAYLOAD.items() if k != "question"}
        resp = client.post("/rag/evaluate", json=payload)
        assert resp.status_code == 422

    def test_empty_contexts_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "contexts": []}
        resp = client.post("/rag/evaluate", json=payload)
        assert resp.status_code == 422
