"""Use Case 1: Prompt Management with Templates and Output Parsing."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import PromptRequest
from ..shared.session_store import LC_MODEL

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/prompt")
def lc_prompt(req: PromptRequest):
    """Demonstrate LangChain prompt templates and StrOutputParser.

    Shows how ChatPromptTemplate composes a system + human message pair,
    and how piping through StrOutputParser extracts the string response.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {role}. Answer in under 2 sentences."),
                ("human", "{question}"),
            ]
        )
        answer = (template | chat_model | StrOutputParser()).invoke(
            {"role": req.role, "question": req.text}
        )
        return {
            "rendered": f"[system] You are a {req.role}. Answer in under 2 sentences.\n[human] {req.text}",
            "answer": answer,
        }
    except Exception as e:
        logger.error("lc_prompt error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
