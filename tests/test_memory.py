"""Tests for Use Case 4: Conversation Memory."""

import pytest


def test_memory_clear_session(client):
    """Test memory endpoint clears session."""
    # Create a session via POST first
    client.post(
        "/langchain/memory",
        json={"session_id": "test-clear", "message": "Test message"},
    )

    # Clear it
    response = client.delete("/langchain/memory/test-clear")
    assert response.status_code == 200
    assert response.json()["cleared"] is True


def test_memory_validation_empty_message(client):
    """Test memory endpoint rejects empty message."""
    response = client.post(
        "/langchain/memory",
        json={"session_id": "test-session", "message": ""},
    )
    assert response.status_code == 422


def test_memory_validation_invalid_session_id(client):
    """Test memory endpoint rejects invalid session_id."""
    response = client.post(
        "/langchain/memory",
        json={"session_id": "invalid@session!", "message": "Test"},
    )
    assert response.status_code == 422


def test_memory_validation_session_id_too_long(client):
    """Test memory endpoint rejects oversized session_id."""
    response = client.post(
        "/langchain/memory",
        json={"session_id": "x" * 101, "message": "Test"},
    )
    assert response.status_code == 422
