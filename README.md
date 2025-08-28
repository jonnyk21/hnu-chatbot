# HNU Chatbot (RAG)

A minimal Retrieval-Augmented Generation (RAG) app for hnu.de content.

- Backend: FastAPI (`hnu-min/app/main.py`), retrieval + generation (`hnu-min/app/rag.py`)
- Ingestion: Crawl4AI-based crawler + ChromaDB upsert (`hnu-min/app/ingest.py`)
- UI: Streamlit chat (`hnu-min/streamlit_app.py`)
- Vector store: ChromaDB (persistent on disk)

## Features
- Single Chroma collection enforced on refresh (always cleared and recreated)
- Sitemap-driven crawl with content filtering and chunking
- OpenAI embeddings + responses API for answers with citations
- Streamlit chat UI consuming the FastAPI backend

## Requirements
- Python 3.12+
- An OpenAI API key

Install Python deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r hnu-min/requirements.txt
```

## Environment configuration
Create `hnu-min/.env` (see `hnu-min/.env.example`). Keys:

- OPENAI_API_KEY=sk-...
- GENERATION_MODEL=gpt-5-nano
- EMBEDDING_MODEL=text-embedding-3-small
- SITEMAP_URL=https://www.hnu.de/sitemap.xml
- CRAWL_START_URL=https://www.hnu.de/
- CHROMA_DIR=./data/chroma  # default; ensure this path is writable
- CHROMA_COLLECTION=hnu     # the only collection used
- ADMIN_TOKEN=change-me     # bearer token for /refresh
- TOP_K=5
- Optional crawl tuning:
  - CRAWL_MAX_URLS, CRAWL_BATCH_SIZE, CRAWL_CONCURRENCY, CRAWL_PAGE_TIMEOUT_MS, CRAWL_WAIT_UNTIL, CRAWL_WAIT_FOR_IMAGES, CRAWL_DELAY_BEFORE_RETURN, CRAWL_CACHE_MODE

Notes:
- `CHROMA_DIR` defaults to an absolute `data/chroma` under the repo to keep a unified store; override if needed.
- `CHROMA_OPENAI_API_KEY` is set automatically from `OPENAI_API_KEY` for Chroma embeddings.

### Tracing (optional)
- Set `LANGSMITH_TRACING=true` to enable tracing to LangSmith.
- Set `LANGSMITH_API_KEY=ls_...` and optionally `LANGSMITH_PROJECT` (defaults to `default`).
- Optional: `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`.
- Tracing is integrated via `@traceable` decorators in `hnu-min/app/ingest.py` and `hnu-min/app/rag.py`, and the OpenAI client is wrapped with `wrap_openai` when available. When tracing is off or LangSmith isn't installed, behavior is unchanged.

## Run the services
Backend (FastAPI):

```bash
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir hnu-min
```

Streamlit UI:

```bash
./.venv/bin/streamlit run ./hnu-min/streamlit_app.py
```

## API
- POST /chat
  - Body: `{"query": "...", "top_k": 5}`
  - Returns: `{ "answer": str, "citations": [{"url": str, "title": str}] }`
- POST /refresh (requires `Authorization: Bearer <ADMIN_TOKEN>`)
  - Always clears and recreates the single configured collection, then re-ingests.
  - Query param `clear` is ignored (clearing is unconditional).
- GET /healthz

## Ingestion & refresh behavior
- `POST /refresh` calls `refresh()` in `hnu-min/app/ingest.py`.
- Refresh logic:
  1) Purge all collections except `CHROMA_COLLECTION`.
  2) Delete `CHROMA_COLLECTION` if it exists.
  3) Re-create it empty.
  4) Crawl sitemap pages and upsert processed chunks.
- After refresh, only one collection exists.

## Architecture overview
- `hnu-min/app/ingest.py` — sitemap discovery, Crawl4AI run configs, cleaning, chunking, Chroma upserts, and enforced single-collection refresh.
- `hnu-min/app/rag.py` — aligns `CHROMA_DIR` with ingest, retrieves from the configured collection, robustly refreshes collection handle after a refresh, and generates answers via OpenAI Responses API.
- `hnu-min/app/main.py` — FastAPI endpoints: `/chat`, `/refresh`, `/healthz`.
- `hnu-min/streamlit_app.py` — Streamlit-based chat client to the backend.
- `hnu-min/scripts/audit_chroma.py` — helper to inspect Chroma collections locally.

## Troubleshooting
- Permission denied (Chroma): ensure `CHROMA_DIR` points to a writable path.
- GET /chat 405: expected; use POST.
- Streamlit warnings about ScriptRunContext: run via `streamlit run`, not `python`.
- NotFoundError (collection missing): the backend re-fetches the collection handle; if empty, run `/refresh`.
- Crawl timeouts/transport issues: lower `CRAWL_BATCH_SIZE`, reduce `CRAWL_CONCURRENCY`, or increase `CRAWL_PAGE_TIMEOUT_MS` in `.env`.

## Security
- Do not commit real API keys.
- `/refresh` is protected by `ADMIN_TOKEN` bearer auth. Keep it secret.

## License
MIT (or your preferred license)
