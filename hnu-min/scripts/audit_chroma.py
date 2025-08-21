#!/usr/bin/env python3
"""
Audit ChromaDB collection for RAG data quality.

Checks:
- Text quality: readability vs boilerplate, symbol ratios
- Chunkability: presence of structure markers, newlines, chunk length ranges
- Embedding-friendliness: length bounds, density
- Metadata usability: url/title/timestamp and optional fields
- Duplicates: normalized hash collisions in sample

Usage:
  python hnu-min/scripts/audit_chroma.py --sample 1000 --batch 500

Runs read-only. Requires chromadb installed and the same CHROMA_DIR as the app.
"""
import os
import re
import sys
import json
import math
import hashlib
import argparse
from collections import Counter, defaultdict

try:
    import chromadb
except Exception as e:
    print("ERROR: chromadb not installed. Activate your venv or run inside the container.")
    sys.exit(1)

# ---------- Config ----------
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "hnu")
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "../../data/chroma")

# Heuristics and patterns
MENU_PATTERNS = [
    r"\bHauptnavigation\b", r"\bServicenavigation\b", r"\bQuicklinks\b",
    r"\bCookie[- ]Hinweis\b", r"\bDiese Website verwendet Cookies\b",
    r"\bThis website uses cookies\b", r"\bMenü\b",
]
EXTRANEOUS_PATTERNS = [
    r"\*\*\(externer Link, öffnet neues Fenster\)\*\*",
    r"\*\*\(external link, opens new window\)\*\*",
]
# Minimal EN/DE stopwords to estimate semantic density
STOPWORDS = set(
    "the a an and or of in to for with on at by from as is are was were be been being it this that these those you we they he she es der die das und zu den dem ist im nicht ein eine einer einem einer".split()
)

def normalize_for_hash(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    # remove most punctuation (basic ASCII class only)
    t = re.sub(r"[^\w\s]", "", t)
    return t

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def text_metrics(doc: str) -> dict:
    chars = len(doc)
    words = doc.split()
    n_words = len(words)
    if chars == 0:
        return {
            "chars": 0, "words": 0, "alpha_ratio": 0.0,
            "digit_ratio": 0.0, "symbol_ratio": 0.0,
            "stopword_ratio": 0.0, "has_newlines": False,
        }
    alpha = sum(ch.isalpha() for ch in doc)
    digit = sum(ch.isdigit() for ch in doc)
    symbol = sum((not ch.isalnum()) and (not ch.isspace()) for ch in doc)
    stop = sum(1 for w in words if w.lower() in STOPWORDS)
    return {
        "chars": chars,
        "words": n_words,
        "alpha_ratio": alpha / chars,
        "digit_ratio": digit / chars,
        "symbol_ratio": symbol / chars,
        "stopword_ratio": (stop / n_words) if n_words else 0.0,
        "has_newlines": ("\n" in doc),
    }

def quality_flags(doc: str) -> dict:
    flags = {}
    m = text_metrics(doc)
    # Embedding-friendly length heuristics (words, not tokens): target 150-400 words
    flags["too_short"] = m["words"] < 80
    flags["too_long"] = m["words"] > 450
    # Symbol-heavy text likely noisy
    flags["symbol_heavy"] = m["symbol_ratio"] > 0.08
    # Very low alpha density is suspicious
    flags["low_alpha_density"] = m["alpha_ratio"] < 0.65
    # Boilerplate hints still present
    flags["boilerplate_terms"] = any(re.search(p, doc, flags=re.IGNORECASE) for p in MENU_PATTERNS + EXTRANEOUS_PATTERNS)
    # Chunkability: absence of newlines suggests we lost structure
    flags["no_structure_markers"] = not m["has_newlines"]
    return flags

def score_flags(flags: dict) -> int:
    # higher is worse
    score = 0
    score += 2 if flags.get("too_long") else 0
    score += 2 if flags.get("too_short") else 0
    score += 1 if flags.get("symbol_heavy") else 0
    score += 1 if flags.get("low_alpha_density") else 0
    score += 2 if flags.get("boilerplate_terms") else 0
    score += 1 if flags.get("no_structure_markers") else 0
    return score

def analyze_collection(sample:int, batch:int, collection_name:str, chroma_dir:str):
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_or_create_collection(collection_name)

    try:
        total = col.count()
    except Exception:
        total = None

    print(f"Collection: {collection_name}  Path: {chroma_dir}")
    print(f"Total docs: {total}")

    fetched = 0
    offset = 0
    batch = max(1, batch)
    sample = sample if sample>0 else (total or 1000)

    dup_counter = Counter()
    problems = []
    url_counts = Counter()
    title_missing = 0
    meta_field_freq = Counter()

    def process(ids, docs, metas):
        nonlocal fetched, title_missing
        for i, doc in enumerate(docs or []):
            meta = (metas or [{}])[i] or {}
            url = meta.get("url", "")
            title = meta.get("title", "")
            url_counts[url] += 1
            if not title:
                title_missing += 1
            for k in meta.keys():
                meta_field_freq[k] += 1
            norm = normalize_for_hash(doc)
            dup_counter[sha1(norm)] += 1
            flags = quality_flags(doc)
            score = score_flags(flags)
            if score:
                problems.append({
                    "id": ids[i],
                    "score": score,
                    "flags": {k:v for k,v in flags.items() if v},
                    "url": url,
                    "title": title[:120],
                    "words": len(doc.split()),
                })
            fetched += 1

    # Try paged get with offset; fallback to single get
    try:
        while fetched < sample:
            take = min(batch, sample - fetched)
            got = col.get(limit=take, offset=offset, include=["documents","metadatas"])  # type: ignore
            if not got or not got.get("ids"):
                break
            process(got.get("ids", []), got.get("documents", []), got.get("metadatas", []))
            offset += len(got["ids"])  # type: ignore
    except Exception:
        got = col.get(limit=sample, include=["documents","metadatas"])  # type: ignore
        process(got.get("ids", []), got.get("documents", []), got.get("metadatas", []))

    # Summaries
    dup_total = sum(c-1 for c in dup_counter.values() if c>1)
    unique = sum(1 for c in dup_counter.values() if c>=1)

    print("\n=== Summary ===")
    print(f"Sampled: {fetched}  Duplicates (extra copies): {dup_total}  Unique hashes: {unique}")
    if fetched:
        print(f"Title missing in {title_missing}/{fetched} ({title_missing*100.0/fetched:.1f}%)")

    print("\nTop meta fields (presence counts):")
    for k, c in meta_field_freq.most_common(15):
        print(f"  {k}: {c}")

    print("\nTop URLs by chunk count:")
    for u, c in url_counts.most_common(10):
        print(f"  {c:4d}  {u}")

    # Worst offenders by score
    problems.sort(key=lambda x: (-x["score"], -x["words"]))
    print("\nWorst 15 chunks (higher score = more issues):")
    for p in problems[:15]:
        print(json.dumps(p, ensure_ascii=False))

    # Bucket by length for embedding-friendliness
    buckets = defaultdict(int)
    for h, c in dup_counter.items():
        # can't recover words here; recompute quickly is heavy. Skip.
        pass

    print("\nHeuristic guidance:")
    print("- Aim for 150-400 words per chunk; too-long or too-short chunks harm retrieval.")
    print("- Preserve some structure markers (newlines, headings) to improve section-aware chunking.")
    print("- Ensure metadata includes url, title, and ideally section_title, heading_path, chunk_index, lastmod.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1000, help="Number of docs to sample")
    ap.add_argument("--batch", type=int, default=500, help="Batch size for paging")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--chroma", default=DEFAULT_CHROMA_DIR)
    args = ap.parse_args()

    analyze_collection(args.sample, args.batch, args.collection, args.chroma)

if __name__ == "__main__":
    main()
