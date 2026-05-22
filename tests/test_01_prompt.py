"""Tests for Use Case 1: Prompt Management."""

import pytest


def test_prompt_validation_empty_role(client):
    """Test prompt endpoint rejects empty role."""
    response = client.post(
        "/langchain/prompt",
        json={"role": "", "text": "What is a list comprehension?"},
    )
    assert response.status_code == 422


def test_prompt_validation_empty_text(client):
    """Test prompt endpoint rejects empty text."""
    response = client.post(
        "/langchain/prompt",
        json={"role": "Python expert", "text": ""},
    )
    assert response.status_code == 422


def test_prompt_validation_text_too_long(client):
    """Test prompt endpoint rejects oversized text."""
    response = client.post(
        "/langchain/prompt",
        json={"role": "Python expert", "text": "x" * 2001},
    )
    assert response.status_code == 422


def test_prompt_validation_role_too_long(client):
    """Test prompt endpoint rejects oversized role."""
    response = client.post(
        "/langchain/prompt",
        json={"role": "x" * 201, "text": "What is a list comprehension?"},
    )
    assert response.status_code == 422


def test_prompt_endpoint_exists(client):
    """Test prompt endpoint is callable (without checking actual LLM output)."""
    response = client.post(
        "/langchain/prompt",
        json={"role": "expert", "text": "test"},
    )
    # Endpoint should respond (either success with valid API key, or error without it)
    assert response.status_code in [200, 500]
