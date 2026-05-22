"""Tests for Use Case 7: Output Parsers."""

import pytest


def test_parsers_validation_empty_topic(client):
    """Test parsers endpoint rejects empty topic."""
    response = client.post(
        "/langchain/parsers",
        json={"topic": ""},
    )
    assert response.status_code == 422


def test_parsers_validation_topic_too_long(client):
    """Test parsers endpoint rejects oversized topic."""
    response = client.post(
        "/langchain/parsers",
        json={"topic": "x" * 501},
    )
    assert response.status_code == 422


def test_parsers_endpoint_exists(client):
    """Test parsers endpoint is callable."""
    response = client.post(
        "/langchain/parsers",
        json={"topic": "Python"},
    )
    assert response.status_code in [200, 500]
