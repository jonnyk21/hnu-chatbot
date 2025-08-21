import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from .rag import retrieve, generate
from .ingest import refresh

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
TOP_K_ENV = int(os.getenv("TOP_K", "5"))

app = FastAPI(title="HNU RAG (Minimal)")

class ChatIn(BaseModel):
    query: str
    top_k: int | None = None

@app.post("/chat")
def chat(body: ChatIn):
    k = body.top_k or TOP_K_ENV
    ctxs = retrieve(body.query, k)
    return generate(body.query, ctxs)


def _auth(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized")


@app.post("/refresh")
def refresh_index(clear: bool = False, _: None = Depends(_auth)):
    added = refresh(clear_override=clear)
    return {"status": "ok", "cleared": clear, "added_chunks": added}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Minimal monthly scheduler (1st of month 03:00)
scheduler = BackgroundScheduler()
scheduler.add_job(refresh, "cron", day=1, hour=3, minute=0)
scheduler.start()
