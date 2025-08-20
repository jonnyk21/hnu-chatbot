import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
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
collection = client.get_or_create_collection("hnu", embedding_function=ef)
oai = OpenAI(api_key=OPENAI_API_KEY)

def retrieve(query: str, k: int = TOP_K):
    res = collection.query(query_texts=[query], n_results=k, include=["metadatas", "documents"])
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    return [{"text": d, "url": m.get("url",""), "title": m.get("title",""), "ts": m.get("ts")}
            for d,m in zip(docs, metas)]

def generate(query: str, ctxs: List[Dict]) -> Dict:
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i,c in enumerate(ctxs)])
    citations = [{"url": c["url"], "title": c.get("title", "")} for c in ctxs if c.get("url")]
    sys = (
        "You answer ONLY from the provided HNU context. "
        "If unsure or missing, say you couldn't find it on hnu.de. "
        "Keep answers concise and include numbered citations like [1], [2]."
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
