"""Use Case 6: Document Processing and Text Splitting."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..shared.models import TextRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/langchain/documents")
def lc_documents(req: TextRequest):
    """Demonstrate CharacterTextSplitter vs RecursiveCharacterTextSplitter.

    Both splitters use chunk_size=200 and chunk_overlap=20 so the difference
    in resulting chunks illustrates how the splitting strategy affects output.
    """
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

        doc = Document(page_content=req.text)
        char_split = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n\n")
        rec_split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        return {
            "total_chars": len(req.text),
            "char_chunks": [
                {"i": i + 1, "chars": len(c.page_content), "text": c.page_content}
                for i, c in enumerate(char_split.split_documents([doc]))
            ],
            "rec_chunks": [
                {"i": i + 1, "chars": len(c.page_content), "text": c.page_content}
                for i, c in enumerate(rec_split.split_documents([doc]))
            ],
        }
    except Exception as e:
        logger.error("lc_documents error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
