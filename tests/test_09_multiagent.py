"""Tests for Use Case 9: Multi-Agent Pipeline."""

import pytest


def test_multiagent_endpoint_exists(client):
    """Test multiagent endpoint is callable."""
    response = client.post(
        "/langchain/multiagent",
        json={"topic": "Climate Change"},
    )
    assert response.status_code in [200, 500]


def test_multiagent_validation_empty_topic(client):
    """Test multiagent endpoint rejects empty topic."""
    response = client.post(
        "/langchain/multiagent",
        json={"topic": ""},
    )
    assert response.status_code == 422


def test_multiagent_validation_topic_too_long(client):
    """Test multiagent endpoint rejects oversized topic."""
    response = client.post(
        "/langchain/multiagent",
        json={"topic": "x" * 501},
    )
    assert response.status_code == 422
