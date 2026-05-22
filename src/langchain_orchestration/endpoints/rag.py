"""Use Case 3: Retrieval Augmented Generation (RAG) with FAISS vectorstore."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import QuestionRequest
from ..shared.session_store import LC_MODEL
from ..shared.vectorstore import get_lc_vectorstore

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/rag")
def lc_rag(req: QuestionRequest):
    """Demonstrate LangChain RAG with a fixed in-memory FAISS vectorstore.

    Uses a hardcoded 7-document AI/ML knowledge base. Shows how LangChain's
    retriever interface abstracts the underlying vector store.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        vectorstore = get_lc_vectorstore()
        chunks = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(req.question)
        context = "\n".join(f"- {d.page_content}" for d in chunks)
        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        answer = (
            ChatPromptTemplate.from_template(
                "Use ONLY the context to answer.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nOne sentence answer:"
            )
            | chat_model
            | StrOutputParser()
        ).invoke({"ctx": context, "q": req.question})
        return {"chunks": [d.page_content for d in chunks], "answer": answer}
    except Exception as e:
        logger.error("lc_rag error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
