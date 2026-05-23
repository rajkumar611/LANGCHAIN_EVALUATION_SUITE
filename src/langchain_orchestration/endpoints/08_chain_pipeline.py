"""Use Case 8: Sequential Chain Pipeline - Researcher and Blog Writer."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import TopicRequest
from ..shared.session_store import LC_MODEL

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/chain-pipeline")
def lc_chain_pipeline(req: TopicRequest):
    """Demonstrate a sequential chain pipeline: researcher → blog writer.

    Two sequential LLM calls with different system prompts simulate specialized
    roles handing off work. The researcher produces facts; the writer crafts prose.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=1024)

        research = chat_model.invoke(
            [
                SystemMessage(
                    "You are a research assistant. Given a topic, return 5 concise bullet-point facts."
                ),
                HumanMessage(req.topic),
            ]
        ).content

        blog = chat_model.invoke(
            [
                SystemMessage(
                    "You are a blog writer. Write a short 3-paragraph blog post (max 150 words) from the research notes provided. Use a friendly tone."
                ),
                HumanMessage(f"Research notes:\n{research}"),
            ]
        ).content

        return {"research": research, "blog": blog}
    except Exception as e:
        logger.error("lc_chain_pipeline error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
