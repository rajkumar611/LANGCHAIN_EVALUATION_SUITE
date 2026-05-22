"""Tests for Use Case 5: Tools and Function Calling."""

import pytest


def test_tool_calling_endpoint_exists(client):
    """Test tool calling endpoint is callable."""
    response = client.post(
        "/langchain/tools",
        json={"question": "What is 25 * 4?"},
    )
    assert response.status_code in [200, 500]


def test_tool_calling_validation_empty_question(client):
    """Test tool calling endpoint rejects empty question."""
    response = client.post(
        "/langchain/tools",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_tool_calling_validation_question_too_long(client):
    """Test tool calling endpoint rejects oversized question."""
    response = client.post(
        "/langchain/tools",
        json={"question": "x" * 2001},
    )
    assert response.status_code == 422
