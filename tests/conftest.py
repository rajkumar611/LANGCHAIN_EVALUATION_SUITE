"""Test configuration and fixtures."""

import os

# Set dummy API key before importing app (pydantic Settings validates at import)
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-key-dummy")

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)
