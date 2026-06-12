"""Chatbot UI backend: FastAPI wrapper around the LangGraph RAG flow. [DD-07]

One JSON endpoint (POST /api/chat) + serves the static chat page at /.
The UI is a thin client: ALL RAG logic stays in rag_graph.py, so the
terminal CLI, the eval harness, and this web UI run the exact same pipeline --
what you demo is what you measured.

Run:  uvicorn src.app:app --reload
Then open http://127.0.0.1:8000
"""
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config
from src.rag_graph import answer_question

app = FastAPI(title="LumenLeaf Support RAG")

STATIC_DIR = config.ROOT / "static"


class ChatRequest(BaseModel):
    # Pydantic validates the request body: must contain a non-empty string.
    question: str = Field(min_length=1, max_length=500)


@app.get("/")
def home():
    """Serve the single-page chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Run one question through the full RAG graph and return everything the
    UI needs -- including transparency fields (score, sources, escalated)
    that make the demo self-explanatory."""
    t0 = time.time()
    result = answer_question(req.question)
    return {
        "answer": result["answer"],
        "escalated": result["escalated"],
        "sources": result.get("sources", []),
        "top_score": round(result.get("top_score", 0.0), 3),
        "latency_s": round(time.time() - t0, 2),
    }
