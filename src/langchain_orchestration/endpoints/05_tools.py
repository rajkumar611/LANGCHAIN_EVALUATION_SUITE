"""Use Case 5: Tools and Function Calling with Calculator, Weather, and Word Count."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import QuestionRequest
from ..shared.session_store import LC_MODEL
from ..shared.tools import make_calculator_tool

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/tools")
def lc_tools(req: QuestionRequest):
    """Demonstrate LangChain tool binding with calculator, weather, and word-count tools.

    Shows how bind_tools registers Python functions as Claude tools, how the
    model decides which tool to call, and how results are fed back for a
    final synthesized answer.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool

        calculator = make_calculator_tool()

        @tool
        def get_weather(city: str) -> str:
            """Return current weather for a city (mock data)."""
            data = {
                "london": "15°C, Cloudy",
                "singapore": "32°C, Humid",
                "new york": "22°C, Sunny",
                "sydney": "19°C, Partly Cloudy",
            }
            return data.get(city.lower(), "No weather data for that city.")

        @tool
        def word_count(text: str) -> str:
            """Count the number of words in a piece of text."""
            return str(len(text.split()))

        tools = [calculator, get_weather, word_count]
        tools_map = {t.name: t for t in tools}
        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=512)
        resp = chat_model.bind_tools(tools).invoke(req.question)
        content = resp.content

        if isinstance(content, str):
            return {"tool_calls": [], "answer": content}

        def _get_field(block, key):
            return block[key] if isinstance(block, dict) else getattr(block, key, None)

        tool_calls = []
        for block in content:
            if _get_field(block, "type") == "tool_use":
                name = _get_field(block, "name")
                inputs = _get_field(block, "input")
                result = tools_map[name].invoke(inputs)
                tool_calls.append({"tool": name, "input": inputs, "result": str(result)})

        if tool_calls:
            ctx = "\n".join(f"{tc['tool']}({tc['input']}) = {tc['result']}" for tc in tool_calls)
            answer = (
                ChatPromptTemplate.from_template(
                    "Question: {q}\n\nTool results:\n{ctx}\n\nAnswer using these results:"
                )
                | chat_model
                | StrOutputParser()
            ).invoke({"q": req.question, "ctx": ctx})
        else:
            texts = [_get_field(b, "text") for b in content if _get_field(b, "type") == "text"]
            answer = " ".join(t for t in texts if t) or "No answer."

        return {"tool_calls": tool_calls, "answer": answer}
    except Exception as e:
        logger.error("lc_tools error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
