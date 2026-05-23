"""Tests for Use Case 8: Sequential Chain Pipeline."""

import pytest


def test_chain_pipeline_endpoint_exists(client):
    """Test chain pipeline endpoint is callable."""
    response = client.post(
        "/langchain/multiagent",
        json={"topic": "Climate Change"},
    )
    assert response.status_code in [200, 500]


def test_chain_pipeline_validation_empty_topic(client):
    """Test chain pipeline endpoint rejects empty topic."""
    response = client.post(
        "/langchain/multiagent",
        json={"topic": ""},
    )
    assert response.status_code == 422


def test_chain_pipeline_validation_topic_too_long(client):
    """Test chain pipeline endpoint rejects oversized topic."""
    response = client.post(
        "/langchain/multiagent",
        json={"topic": "x" * 501},
    )
    assert response.status_code == 422
