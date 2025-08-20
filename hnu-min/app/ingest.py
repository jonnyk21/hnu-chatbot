import os, time, hashlib, asyncio
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Final
from urllib.parse import urlparse, urlsplit, urlunsplit
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    CacheMode,
)
from crawl4ai.content_filter_strategy import PruningContentFilter
import chromadb
from chromadb.utils import embedding_functions
from .utils import clean_html, chunk_text
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
START_URL = os.getenv("CRAWL_START_URL", "https://www.hnu.de/")
SITEMAP_URL = os.getenv("SITEMAP_URL", "https://www.hnu.de/sitemap.xml")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_URLS = int(os.getenv("CRAWL_MAX_URLS", "10000"))
BATCH_SIZE = int(os.getenv("CRAWL_BATCH_SIZE", "120"))
CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "16"))
PAGE_TIMEOUT_MS = int(os.getenv("CRAWL_PAGE_TIMEOUT_MS", "30000"))
WAIT_UNTIL = os.getenv("CRAWL_WAIT_UNTIL", "domcontentloaded")
WAIT_FOR_IMAGES = os.getenv("CRAWL_WAIT_FOR_IMAGES", "false").lower() == "true"
DELAY_BEFORE_RETURN = float(os.getenv("CRAWL_DELAY_BEFORE_RETURN", "0"))
# Cache modes vary by version; ensure safe default without touching missing members
# Known in 0.7.x: BYPASS (others may not exist). Default to BYPASS.
CACHE_MODE_NAME = os.getenv("CRAWL_CACHE_MODE", "BYPASS").upper()
_CACHE_MODE = getattr(CacheMode, CACHE_MODE_NAME, None) or getattr(CacheMode, "BYPASS")

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
            return path

CHROMA_DIR = _ensure_chroma_dir(CHROMA_DIR)

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name=EMB_MODEL
)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("hnu", embedding_function=ef)

def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()

def _domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc.split("@").pop().split(":")[0]
    except Exception:
        return url

def _normalize_domain(d: str) -> str:
    d = d.lower()
    if d.startswith("www."):
        return d[4:]
    return d

_EXCLUDED_EXTS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".ico", ".bmp", ".tiff", ".webm", ".mp4", ".mp3", ".wav",
    ".css", ".js", ".json", ".xml", ".rss", ".zip", ".gz", ".rar",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
}

def _normalize_url(u: str) -> str | None:
    try:
        sp = urlsplit(u)
        # must be http(s)
        if sp.scheme not in ("http", "https"):
            return None
        # drop fragments
        sp = sp._replace(fragment="")
        # optional: keep queries but can collapse tracking later
        path = sp.path or "/"
        # unify trailing slash: directories end with '/'
        if not os.path.splitext(path)[1] and not path.endswith("/"):
            path = path + "/"
        sp = sp._replace(path=path)
        return urlunsplit(sp)
    except Exception:
        return None

_BLOCKED_DOMAINS: Final = {
    "campus.hnu.de",
    "intern.hnu.de",
    "ezproxy.hnu.de",
}

def _is_allowed(u: str, root_domain: str) -> bool:
    try:
        cand_domain = _normalize_domain(_domain_from_url(u))
        if cand_domain in _BLOCKED_DOMAINS:
            return False

        base = _normalize_domain(root_domain)
        if not (cand_domain == base or cand_domain.endswith("." + base)):
            return False

        path = urlsplit(u).path or "/"
        ext = os.path.splitext(path)[1].lower()
        if ext in _EXCLUDED_EXTS:
            return False

        return True
    except Exception:
        return False

def _build_run_config() -> CrawlerRunConfig:
    """Primary LLM-friendly config with conservative noise reduction that works across templates.

    Avoids over-constraining selectors; relies on markdown generation + simple excludes.
    """
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.50,
            threshold_type="fixed",
            min_word_threshold=10,
        ),
        options={
            "ignore_links": True,
            # Disable wrapping for cleaner structure; better for chunking
            "body_width": 0,
        },
    )
    return CrawlerRunConfig(
        markdown_generator=md_generator,
        # Global, conservative noise reduction
        remove_overlay_elements=True,
        excluded_tags=["nav", "footer", "aside", "header"],
        remove_forms=True,
        only_text=False,
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_external_images=True,
        # Performance & reliability
        semaphore_count=CONCURRENCY,
        page_timeout=PAGE_TIMEOUT_MS,
        cache_mode=_CACHE_MODE,
        wait_until=WAIT_UNTIL,
        wait_for_images=WAIT_FOR_IMAGES,
        delay_before_return_html=DELAY_BEFORE_RETURN,
        verbose=True,
    )

def _build_permissive_config() -> CrawlerRunConfig:
    """Fallback config: minimal filtering to ensure we never return empty content."""
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "body_width": 120,
        }
    )
    return CrawlerRunConfig(
        markdown_generator=md_generator,
        semaphore_count=CONCURRENCY,
        page_timeout=PAGE_TIMEOUT_MS,
        cache_mode=_CACHE_MODE,
        wait_until=WAIT_UNTIL,
        wait_for_images=WAIT_FOR_IMAGES,
        delay_before_return_html=DELAY_BEFORE_RETURN,
        verbose=True,
    )

def _extract_clean_text(result) -> str:
    """Return cleaned text from a CrawlResult, preferring v2/raw/fit markdown and stripping boilerplate."""
    # Prefer markdown_v2 if available (already processed by generator)
    md = getattr(result, "markdown_v2", None)
    markdown_obj = getattr(result, "markdown", None)
    if markdown_obj is not None:
        # Prefer raw_markdown first; fit_markdown can be overly short without filters
        md = md or getattr(markdown_obj, "raw_markdown", None) or getattr(markdown_obj, "fit_markdown", None)
    md = md or getattr(result, "extracted_content", None) or getattr(result, "content", None) or ""
    text = clean_html(md)
    text = _strip_boilerplate(text)
    # Normalize mid-word line breaks: convert "a\nb" -> "a b" while keeping paragraph breaks intact
    text = re.sub(r"(\S)\n(\S)", r"\1 \2", text)
    # Remove soft hyphen and zero-width characters that cause mid-word artifacts
    text = re.sub(r"[\u00AD\u200B\u200C\u200D\u2060]", "", text)
    return text

"""Unified global settings; no per-section markdown generator needed."""

def _strip_boilerplate(text: str) -> str:
    """Remove common navigation/cookie/menu boilerplate phrases from HNU pages.

    Conservative regex-based removal to reduce noise while retaining content.
    """
    import re
    t = text
    # Remove lines starting with navigation hints
    patterns = [
        r"^Springe direkt\s+zu:.*$",
        r"^Hauptnavigation.*$",
        r"^Servicenavigation.*$",
        r"^Quicklinks.*$",
        r"^Login\s*$",
        # Header/menu/breadcrumbs/logo
        r"^\s*Menü\s*$",
        r"^\s*Zurück zur vorherigen Seite\s*$",
        r"^\s*Zurück\s*$",
        r"^\s*\[zur Startseite\]\([^)]+\)\s*$",
        r"^\s*Sie sind hier:.*$",
        r"^\s*Startseite\s*/.*$",
        # Cookie notices (DE/EN variants)
        r"^\s*Cookie[- ]Hinweis.*$",
        r"^\s*Diese Website verwendet Cookies.*$",
        r"^\s*This website uses cookies.*$",
        # Lines that are just asterisks blocks like '* * *'
        r"^\s*(\*\s*){2,}\s*$",
        # Specific logo image line (avoid broad inline deletion)
        r"^\s*!\[zur Startseite\]\([^)]+logo-main\.svg\)\s*$",
        # Decorative header line with 'Menü' surrounded by asterisk bullets
        r"^\s*(\*\s*){1,}.*\bMenü\b.*$",
    ]
    for pat in patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE | re.MULTILINE)
    # Trim decorative leading asterisk runs (e.g., "* * * *  # Heading")
    t = re.sub(r"^(?:\*\s*){2,}", "", t, flags=re.MULTILINE)
    # Remove inline breadcrumb phrase while preserving rest of the line
    t = re.sub(r"Zurück zur vorherigen Seite", "", t, flags=re.IGNORECASE)
    # Drop common repeated external link disclaimers cluttering text
    t = re.sub(r"\*\*\(externer Link, öffnet neues Fenster\)\*\*", "", t)
    t = re.sub(r"\*\*\(external link, opens new window\)\*\*", "", t, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

async def _discover_urls(start_url: str, max_urls: int = MAX_URLS) -> list[str]:
    """Discover crawl targets from the site's sitemap.xml only."""
    domain = _domain_from_url(start_url)

    def _http_get_bytes(url: str) -> bytes | None:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception:
            return None

    def _parse_sitemap_bytes(data: bytes) -> tuple[list[str], list[str]]:
        """Return (sitemap_urls, page_urls) from a sitemap document."""
        try:
            root = ET.fromstring(data)
        except Exception:
            return ([], [])
        ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        smaps: list[str] = []
        pages: list[str] = []
        if root.tag.endswith("sitemapindex"):
            for sm in root.findall(f"{ns}sitemap"):
                loc = sm.find(f"{ns}loc")
                if loc is not None and loc.text:
                    smaps.append(loc.text.strip())
        elif root.tag.endswith("urlset"):
            for u in root.findall(f"{ns}url"):
                loc = u.find(f"{ns}loc")
                if loc is not None and loc.text:
                    pages.append(loc.text.strip())
        return (smaps, pages)

    # BFS over sitemap index(s)
    to_visit = [SITEMAP_URL]
    visited = set()
    collected_pages: list[str] = []
    while to_visit and len(collected_pages) < max_urls:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        data = await asyncio.to_thread(_http_get_bytes, current)
        if not data:
            continue
        smaps, pages = _parse_sitemap_bytes(data)
        # Add nested sitemaps
        for sm in smaps:
            if sm not in visited:
                to_visit.append(sm)
        collected_pages.extend(pages)

    # Normalize and filter to in-domain HTML-like pages
    normalized: list[str] = []
    seen = set()
    for u in collected_pages:
        nu = _normalize_url(u)
        if not nu:
            continue
        if not _is_allowed(nu, domain):
            continue
        if nu in seen:
            continue
        seen.add(nu)
        normalized.append(nu)

    # Ensure start URL is included
    if start_url not in normalized:
        normalized.insert(0, _normalize_url(start_url) or start_url)

    print(f"[ingest] discovered via sitemap: raw={len(collected_pages)} normalized={len(normalized)} (cap {max_urls})")
    return normalized[:max_urls]

async def _crawl_many(urls: list[str]) -> list:
    primary = _build_run_config()
    results_all = []
    browser_cfg = BrowserConfig(
        headless=True,
        verbose=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for i in range(0, len(urls), BATCH_SIZE):
            batch = urls[i : i + BATCH_SIZE]
            print(f"[ingest] crawling batch {i//BATCH_SIZE + 1}: {len(batch)} urls")
            res = await crawler.arun_many(urls=batch, config=primary)
            results_all.extend(res)
    return results_all

def _upsert_batches(chunks: list[str], ids: list[str], metas: list[dict], batch_size: int = 100):
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            documents=chunks[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )

def full_ingest(max_urls: int = MAX_URLS) -> int:
    async def _run():
        urls = await _discover_urls(START_URL, max_urls=max_urls)
        return await _crawl_many(urls)

    results = asyncio.run(_run())
    added = 0

    docs: list[str] = []
    ids: list[str] = []
    metas: list[dict] = []

    for r in results or []:
        if not getattr(r, "success", False):
            continue
        url = getattr(r, "url", None) or getattr(r, "final_url", START_URL)
        metadata = getattr(r, "metadata", None)
        title = (metadata.get("title") if metadata else None) or getattr(r, "title", "") or getattr(r, "page_title", "")
        text = _extract_clean_text(r)
        if len(text) < 200:
            continue

        parts = chunk_text(text)
        docs.extend(parts)
        ids.extend([f"{_doc_id(url)}::{i}" for i, _ in enumerate(parts)])
        metas.extend([{ "url": url, "title": title, "ts": int(time.time()) } for _ in parts])
        added += len(parts)

        # Periodic upsert to cap memory
        if len(docs) >= 300:
            _upsert_batches(docs, ids, metas)
            docs.clear(); ids.clear(); metas.clear()

    if docs:
        _upsert_batches(docs, ids, metas)

    return added

def refresh() -> int:
    # Simple: just call full_ingest() for MVP
    return full_ingest()
