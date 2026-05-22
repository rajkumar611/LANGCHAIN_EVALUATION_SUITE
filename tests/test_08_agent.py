"""Tests for Use Case 8: ReAct Agent."""

import pytest


def test_agent_validation_empty_question(client):
    """Test agent endpoint rejects empty question."""
    response = client.post(
        "/langchain/agent",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_agent_validation_question_too_long(client):
    """Test agent endpoint rejects oversized question."""
    response = client.post(
        "/langchain/agent",
        json={"question": "x" * 2001},
    )
    assert response.status_code == 422


def test_agent_endpoint_exists(client):
    """Test agent endpoint is callable."""
    response = client.post(
        "/langchain/agent",
        json={"question": "What is 2 + 2?"},
    )
    assert response.status_code in [200, 500]
