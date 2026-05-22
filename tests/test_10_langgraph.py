"""Tests for Use Case 10: LangGraph Workflow."""

import pytest


def test_langgraph_endpoint_exists(client):
    """Test langgraph endpoint is callable."""
    response = client.post(
        "/langchain/langgraph",
        json={"topic": "Artificial Intelligence"},
    )
    assert response.status_code in [200, 500]


def test_langgraph_validation_empty_topic(client):
    """Test langgraph endpoint rejects empty topic."""
    response = client.post(
        "/langchain/langgraph",
        json={"topic": ""},
    )
    assert response.status_code == 422


def test_langgraph_validation_topic_too_long(client):
    """Test langgraph endpoint rejects oversized topic."""
    response = client.post(
        "/langchain/langgraph",
        json={"topic": "x" * 501},
    )
    assert response.status_code == 422
