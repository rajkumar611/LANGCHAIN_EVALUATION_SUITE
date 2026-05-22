"""Tests for Use Case 3: RAG (Retrieval Augmented Generation)."""

import pytest


def test_rag_validation_empty_question(client):
    """Test RAG endpoint rejects empty question."""
    response = client.post(
        "/langchain/rag",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_rag_validation_question_too_long(client):
    """Test RAG endpoint rejects oversized question."""
    response = client.post(
        "/langchain/rag",
        json={"question": "x" * 2001},
    )
    assert response.status_code == 422
