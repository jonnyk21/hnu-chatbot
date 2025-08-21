import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)
_DEFAULT_CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma"))
CHROMA_DIR = os.getenv("CHROMA_DIR", _DEFAULT_CHROMA_DIR)
TOP_K = int(os.getenv("TOP_K", "5"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEN_MODEL = os.getenv("GENERATION_MODEL", "gpt-5-nano")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GEN_TEMPERATURE = os.getenv("GEN_TEMPERATURE")

# Ensure Chromadb's OpenAI embedding function sees an API key
if not os.getenv("CHROMA_OPENAI_API_KEY") and OPENAI_API_KEY:
    os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY

# Ensure the Chroma persistence directory exists; if not writable, fallback to local path
def _ensure_chroma_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        fallback = "./data/chroma"
        try:
            os.makedirs(fallback, exist_ok=True)
            return fallback
        except Exception:
            return path  # let Chroma raise a clearer error

CHROMA_DIR = _ensure_chroma_dir(CHROMA_DIR)

client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name=EMB_MODEL
)
# Use a configurable collection name to stay consistent with ingestion
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hnu")
oai = OpenAI(api_key=OPENAI_API_KEY)

def _get_collection(name: str | None = None):
    """Return a collection handle by name, creating it if needed."""
    return client.get_or_create_collection(name or COLLECTION_NAME, embedding_function=ef)

def retrieve(query: str, k: int = TOP_K, collection_name: str | None = None):
    # Retrieve more candidates for reranking and deduplication
    candidate_k = min(k * 3, 50)
    coll = _get_collection(collection_name)
    try:
        res = coll.query(
            query_texts=[query], 
            n_results=candidate_k, 
            include=["metadatas", "documents", "distances"]
        )
    except chromadb.errors.NotFoundError:
        # Collection might have been deleted or path changed; refresh handle and retry once
        coll = _get_collection(collection_name)
        res = coll.query(
            query_texts=[query], 
            n_results=candidate_k, 
            include=["metadatas", "documents", "distances"]
        )
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    distances = res["distances"][0] if res.get("distances") else []
    
    # Build candidates with scores
    candidates = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        score = 1.0 - (distances[i] if i < len(distances) else 0.5)  # Convert distance to similarity
        candidates.append({
            "text": doc,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "ts": meta.get("ts"),
            "chunk_index": meta.get("chunk_index", 0),
            "content_length": meta.get("content_length", len(doc.split())),
            "score": score
        })
    
    # Simple MMR-style reranking: select diverse, high-scoring chunks
    selected = []
    seen_urls = set()
    total_words = 0
    max_words = 1200  # Token budget (~1500 tokens)
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    for cand in candidates:
        if len(selected) >= k:
            break
        
        url = cand["url"]
        words = cand["content_length"]
        
        # Skip if we already have 2+ chunks from this URL (diversity)
        url_count = sum(1 for s in selected if s["url"] == url)
        if url_count >= 2:
            continue
            
        # Skip if adding this would exceed word budget
        if total_words + words > max_words and selected:
            continue
            
        selected.append(cand)
        total_words += words
        seen_urls.add(url)
    
    return selected

def generate(query: str, ctxs: List[Dict]) -> Dict:
    # Build context with better formatting and source info
    context_parts = []
    for i, c in enumerate(ctxs):
        chunk_info = f" (chunk {c.get('chunk_index', 0)+1}/{c.get('chunk_count', 1)})" if c.get('chunk_count', 1) > 1 else ""
        context_parts.append(f"[{i+1}] {c['text']}{chunk_info}")
    
    context = "\n\n".join(context_parts)
    
    # Deduplicate citations by URL
    seen_urls = set()
    citations = []
    for c in ctxs:
        url = c.get("url")
        if url and url not in seen_urls:
            citations.append({"url": url, "title": c.get("title", "")})
            seen_urls.add(url)
    
    sys = (
        "You answer ONLY from the provided HNU context. "
        "If unsure or missing, say you couldn't find it on hnu.de. "
        "Keep answers concise and include numbered citations like [1], [2]. "
        "When multiple chunks are from the same page, synthesize the information."
    )
    user = f"Question: {query}\n\nContext:\n{context}"

    # Use Responses API for compatibility with latest models (e.g., GPT‑5)
    # Concatenate system guidance and user/input into a single input string.
    args = {
        "model": GEN_MODEL,
        "input": f"System: {sys}\n\nUser: {user}",
    }
    # Only include temperature if explicitly configured; some models reject it
    if GEN_TEMPERATURE is not None:
        try:
            args["temperature"] = float(GEN_TEMPERATURE)
        except ValueError:
            pass

    try:
        resp = oai.responses.create(**args)
    except BadRequestError as e:
        # Retry once without temperature if the model doesn't support it
        if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
            args.pop("temperature", None)
            resp = oai.responses.create(**args)
        else:
            raise
    answer = getattr(resp, "output_text", None)
    if not answer:
        # Fallback: attempt to extract from content blocks if output_text missing
        try:
            answer = "".join([b.get("text", "") for c in getattr(resp, "output", []) for b in getattr(c, "content", [])])
        except Exception:
            answer = ""
    return {"answer": answer, "citations": citations}
