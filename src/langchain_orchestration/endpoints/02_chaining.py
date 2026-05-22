"""Use Case 2: LLM Chaining - Sequential 3-step transformation."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import TextRequest
from ..shared.session_store import LC_MODEL

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/chaining")
def lc_chaining(req: TextRequest):
    """Demonstrate sequential LangChain chaining: translate → summarise → JSON.

    Three independent chains run in sequence, each consuming the previous
    step's output. Shows how the pipe operator (|) composes LCEL chains.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        str_parser = StrOutputParser()

        step1 = (
            ChatPromptTemplate.from_template(
                "Translate to English. Return ONLY the translation.\n\n{text}"
            )
            | chat_model
            | str_parser
        ).invoke({"text": req.text})

        step2 = (
            ChatPromptTemplate.from_template("Summarise in exactly one sentence:\n\n{t}")
            | chat_model
            | str_parser
        ).invoke({"t": step1})

        step3 = (
            ChatPromptTemplate.from_template('Wrap in JSON: {{"summary": "..."}}\nSummary: {s}')
            | chat_model
            | str_parser
        ).invoke({"s": step2})

        return {
            "steps": [
                {"label": "1 · Translate", "output": step1},
                {"label": "2 · Summarise", "output": step2},
                {"label": "3 · JSON", "output": step3},
            ]
        }
    except Exception as e:
        logger.error("lc_chaining error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
