"""Session management and model configuration for LangChain endpoints."""

from src.config import settings

LC_SESSIONS: dict[str, list] = {}
LC_MODEL = settings.haiku_model
