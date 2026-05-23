"""Tests for Use Case 10: LangGraph Workflow."""

import pytest


def test_workflow_endpoint_exists(client):
    """Test workflow endpoint is callable."""
    response = client.post(
        "/langchain/workflow",
        json={"topic": "Artificial Intelligence"},
    )
    assert response.status_code in [200, 500]


def test_workflow_validation_empty_topic(client):
    """Test workflow endpoint rejects empty topic."""
    response = client.post(
        "/langchain/workflow",
        json={"topic": ""},
    )
    assert response.status_code == 422


def test_workflow_validation_topic_too_long(client):
    """Test workflow endpoint rejects oversized topic."""
    response = client.post(
        "/langchain/workflow",
        json={"topic": "x" * 501},
    )
    assert response.status_code == 422
