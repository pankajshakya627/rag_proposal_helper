from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag import fine_prints, chat_graph

app = FastAPI(title="Proposal-Drafting RAG API 🚀")

class ChatReq(BaseModel):
    question: str

@app.on_event("startup")
def _init():
    app.state.fp   = fine_prints.load()
    app.state.graph = chat_graph.graph

@app.get("/fine-prints")
def get_fp():
    if not app.state.fp:
        raise HTTPException(404, "Fine-prints missing. Run ingest first.")
    return app.state.fp

@app.post("/chat")
def chat(req: ChatReq):
    res = app.state.graph.invoke({"question": req.question})
    return res
