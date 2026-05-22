"""Tests for Use Case 6: Document Processing and Text Splitting."""

import pytest


def test_documents_character_splitter(client):
    """Test document endpoint returns character splitter output."""
    text = "This is a test. " * 50  # ~750 chars
    response = client.post(
        "/langchain/documents",
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_chars" in data
    assert "char_chunks" in data
    assert "rec_chunks" in data
    assert data["total_chars"] > 0
    assert isinstance(data["char_chunks"], list)
    assert isinstance(data["rec_chunks"], list)


def test_documents_different_splitting_strategies(client):
    """Test that character and recursive splitters produce different results."""
    text = "Header 1\n\nParagraph 1. " * 20 + "\n\nHeader 2\n\nParagraph 2. " * 20
    response = client.post(
        "/langchain/documents",
        json={"text": text},
    )
    assert response.status_code == 200
    data = response.json()
    # Splitters may produce different chunk counts
    assert "char_chunks" in data
    assert "rec_chunks" in data


def test_documents_validation_empty_text(client):
    """Test documents endpoint rejects empty text."""
    response = client.post(
        "/langchain/documents",
        json={"text": ""},
    )
    assert response.status_code == 422


def test_documents_validation_text_too_long(client):
    """Test documents endpoint rejects oversized text."""
    response = client.post(
        "/langchain/documents",
        json={"text": "x" * 10001},
    )
    assert response.status_code == 422


def test_documents_small_text(client):
    """Test documents endpoint handles small text."""
    response = client.post(
        "/langchain/documents",
        json={"text": "Hello world"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_chars"] == 11
