"""Use Case 7: Output Parsers - StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import TopicRequest
from ..shared.session_store import LC_MODEL

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/parsers")
def lc_parsers(req: TopicRequest):
    """Demonstrate StrOutputParser, JsonOutputParser, and CommaSeparatedListOutputParser.

    All three parsers receive the same topic but use different format instructions,
    showing how LangChain parsers shape LLM output into typed Python objects.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import (
            CommaSeparatedListOutputParser,
            JsonOutputParser,
            StrOutputParser,
        )
        from langchain_core.prompts import ChatPromptTemplate

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=512)
        list_parser = CommaSeparatedListOutputParser()
        json_parser = JsonOutputParser()

        list_output = (
            ChatPromptTemplate.from_template("List five items related to '{t}'. {fi}").partial(
                fi=list_parser.get_format_instructions()
            )
            | chat_model
            | list_parser
        ).invoke({"t": req.topic})

        json_output = (
            ChatPromptTemplate.from_template(
                "Return JSON about '{t}' with keys: name, description, key_facts (array of 3). {fi}"
            ).partial(fi=json_parser.get_format_instructions())
            | chat_model
            | json_parser
        ).invoke({"t": req.topic})

        str_output = (
            ChatPromptTemplate.from_template("Write one sentence explaining '{t}'.")
            | chat_model
            | StrOutputParser()
        ).invoke({"t": req.topic})

        return {"string_output": str_output, "list_output": list_output, "json_output": json_output}
    except Exception as e:
        logger.error("lc_parsers error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
