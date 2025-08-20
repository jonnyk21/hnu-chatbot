import re
from typing import List

def clean_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

def chunk_text(text: str, tokens: int = 900, overlap: int = 120) -> List[str]:
    # naive splitter by sentence length proxy
    words = text.split()
    step = tokens - overlap
    return [" ".join(words[i:i+tokens]) for i in range(0, len(words), step)]
