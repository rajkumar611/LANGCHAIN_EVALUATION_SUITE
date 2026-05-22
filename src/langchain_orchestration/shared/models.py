"""Pydantic request models for LangChain endpoints."""

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    role: str = Field(default="helpful assistant", min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=2000)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)


class MemoryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
