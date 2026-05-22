"""Tests for Use Case 2: LLM Chaining."""

import pytest


def test_chaining_validation_empty_text(client):
    """Test chaining endpoint rejects empty text."""
    response = client.post(
        "/langchain/chaining",
        json={"text": ""},
    )
    assert response.status_code == 422


def test_chaining_validation_text_too_long(client):
    """Test chaining endpoint rejects oversized text."""
    response = client.post(
        "/langchain/chaining",
        json={"text": "x" * 10001},
    )
    assert response.status_code == 422


def test_chaining_endpoint_exists(client):
    """Test chaining endpoint is callable."""
    response = client.post(
        "/langchain/chaining",
        json={"text": "test message"},
    )
    assert response.status_code in [200, 500]
