"""Use Case 10: LangGraph Workflow - Manager, Researcher, Writer, Reviewer with Conditional Edges."""

import logging
from typing import Literal, TypedDict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import TopicRequest
from ..shared.session_store import LC_MODEL

router = APIRouter()
logger = logging.getLogger(__name__)


class BlogState(TypedDict):
    topic: str
    research: str
    draft: str
    feedback: str
    final: str
    revisions: int


@router.post("/langchain/langgraph")
def lc_langgraph(req: TopicRequest):
    """Demonstrate a LangGraph StateGraph with manager → research → writer → reviewer loop.

    Builds a directed graph with a conditional edge: the reviewer either approves
    the draft or sends it back to the writer for revision (max 2 revisions total).
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import END, StateGraph

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=512)
        str_parser = StrOutputParser()
        log: list[dict] = []

        def manager_node(state: BlogState) -> BlogState:
            log.append(
                {
                    "node": "Manager",
                    "status": "ok",
                    "detail": f"Planning pipeline for: {state['topic']}",
                }
            )
            return state

        def research_node(state: BlogState) -> BlogState:
            output = (
                ChatPromptTemplate.from_template(
                    "Research '{t}'. Return 5 concise bullet-point facts."
                )
                | chat_model
                | str_parser
            ).invoke({"t": state["topic"]})
            log.append({"node": "Research Agent", "status": "ok", "detail": output})
            return {"research": output}

        def writer_node(state: BlogState) -> BlogState:
            feedback_block = f"\n\nFeedback:\n{state['feedback']}" if state.get("feedback") else ""
            output = (
                ChatPromptTemplate.from_template(
                    "Write a 3-paragraph blog post (max 120 words).\n\nResearch:\n{r}{fb}"
                )
                | chat_model
                | str_parser
            ).invoke({"r": state["research"], "fb": feedback_block})
            log.append({"node": "Writer Agent", "status": "ok", "detail": output})
            return {"draft": output, "revisions": state.get("revisions", 0)}

        def reviewer_node(state: BlogState) -> BlogState:
            review = (
                ChatPromptTemplate.from_template(
                    "Review this blog. Reply APPROVED or REVISE: <feedback>\n\nDraft:\n{d}"
                )
                | chat_model
                | str_parser
            ).invoke({"d": state["draft"]})
            approved = review.strip().upper().startswith("APPROVED")
            log.append(
                {
                    "node": "Reviewer Agent",
                    "status": "approved" if approved else "revise",
                    "detail": review,
                }
            )
            if approved:
                return {"final": state["draft"], "feedback": ""}
            return {"feedback": review, "revisions": state.get("revisions", 0) + 1}

        def should_revise(state: BlogState) -> Literal["writer", "end"]:
            if state.get("final") or state.get("revisions", 0) >= 2:
                return "end"
            return "writer"

        graph = StateGraph(BlogState)
        for name, fn in [
            ("manager", manager_node),
            ("research", research_node),
            ("writer", writer_node),
            ("reviewer", reviewer_node),
        ]:
            graph.add_node(name, fn)

        graph.set_entry_point("manager")
        graph.add_edge("manager", "research")
        graph.add_edge("research", "writer")
        graph.add_edge("writer", "reviewer")
        graph.add_conditional_edges("reviewer", should_revise, {"writer": "writer", "end": END})
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "topic": req.topic,
                "research": "",
                "draft": "",
                "feedback": "",
                "final": "",
                "revisions": 0,
            }
        )
        return {
            "log": log,
            "final": result.get("final") or result.get("draft"),
            "revisions": result.get("revisions", 0),
        }
    except Exception as e:
        logger.error("lc_langgraph error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
