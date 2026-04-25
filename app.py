"""
AI Demos — unified FastAPI app
Run via: python main.py
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from rag.routes import router as rag_router
from lc.routes import router as lc_router

app = FastAPI(title="AI Learning Hub")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(rag_router)
app.include_router(lc_router)

@app.get("/")
def root():
    return FileResponse("index.html")
