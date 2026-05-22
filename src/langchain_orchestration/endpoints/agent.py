"""Use Case 8: ReAct Agent with Tool Integration."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import QuestionRequest
from ..shared.session_store import LC_MODEL
from ..shared.tools import make_calculator_tool

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/agent")
def lc_agent_ep(req: QuestionRequest):
    """Demonstrate a ReAct agent via langgraph.prebuilt.create_react_agent.

    The agent has access to calculator, exchange rate, and country info tools.
    It reasons step-by-step, deciding which tools to call before answering.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        calculator = make_calculator_tool()

        @tool
        def get_exchange_rate(currency_pair: str) -> str:
            """Get exchange rate between two currencies. Format: 'USD_SGD'."""
            rates = {"usd_sgd": 1.35, "eur_usd": 1.08, "gbp_usd": 1.27, "inr_usd": 0.012}
            key = currency_pair.lower().replace("/", "_").replace("-", "_")
            parts = key.split("_")
            rate = rates.get(key)
            return (
                f"1 {parts[0].upper()} = {rate} {parts[1].upper()}" if rate else "Rate not found."
            )

        @tool
        def get_country_info(country: str) -> str:
            """Return basic facts about a country. Supports: Singapore, India, USA."""
            info = {
                "singapore": "City-state in Southeast Asia. Population ~6M. Currency: SGD.",
                "india": "South Asian country. Population ~1.4B. Currency: INR.",
                "usa": "North American country. Population ~330M. Currency: USD.",
            }
            return info.get(country.lower(), "Country info not available.")

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=1024)
        agent = create_react_agent(
            chat_model,
            [calculator, get_exchange_rate, get_country_info],
            prompt="You are a helpful assistant. Use tools when needed. Think step by step.",
        )
        result = agent.invoke({"messages": [{"role": "user", "content": req.question}]})

        steps: list[dict] = []
        pending: dict[str, dict] = {}
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    pending[tool_call["id"]] = {
                        "tool": tool_call["name"],
                        "input": tool_call["args"],
                        "result": "",
                    }
            elif isinstance(msg, ToolMessage) and msg.tool_call_id in pending:
                pending[msg.tool_call_id]["result"] = msg.content
                steps.append(pending.pop(msg.tool_call_id))

        return {"steps": steps, "answer": result["messages"][-1].content}
    except Exception as e:
        logger.error("lc_agent error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
