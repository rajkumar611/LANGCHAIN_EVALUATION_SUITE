"""Lazy-loaded FAISS vectorstore singleton for LangChain RAG endpoint."""

EMB_MODEL = "all-MiniLM-L6-v2"

_lc_vectorstore = None


def get_lc_vectorstore():
    """Lazy-load a small fixed FAISS vectorstore with 7 AI/ML facts."""
    global _lc_vectorstore
    if _lc_vectorstore:
        return _lc_vectorstore

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Python is a high-level programming language prized for its readability."
        ),
        Document(
            page_content="LangChain is a framework for building LLM-powered applications with chains and agents."
        ),
        Document(page_content="Vector databases store embeddings and enable fast semantic search."),
        Document(
            page_content="RAG stands for Retrieval Augmented Generation — fetch relevant context, then generate."
        ),
        Document(
            page_content="FAISS is Meta AI's library for fast similarity search over dense vectors."
        ),
        Document(
            page_content="Embeddings convert text to numerical vectors that capture semantic meaning."
        ),
        Document(
            page_content="Chunking splits large documents into smaller pieces before embedding."
        ),
    ]
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    _lc_vectorstore = FAISS.from_documents(docs, emb)
    return _lc_vectorstore
