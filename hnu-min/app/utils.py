import re
from typing import List

def clean_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

def chunk_text(text: str, target_words: int = 300, overlap: int = 50) -> List[str]:
    """Structure-aware chunking that preserves paragraphs and headings.
    
    Args:
        text: Input text to chunk
        target_words: Target chunk size in words (will vary to respect boundaries)
        overlap: Word overlap between chunks
    """
    if not text.strip():
        return []
    
    # Split into paragraphs first (double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        # Fallback: split by single newlines
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_words = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        # If paragraph alone exceeds target, split it
        if para_words > target_words * 1.5:
            # Finish current chunk if any
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_words = 0
            
            # Split large paragraph by sentences
            sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
            temp_chunk = []
            temp_words = 0
            
            for sent in sentences:
                sent_words = len(sent.split())
                if temp_words + sent_words > target_words and temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    # Keep overlap
                    if len(temp_chunk) > 1:
                        temp_chunk = temp_chunk[-1:]
                        temp_words = len(temp_chunk[0].split())
                    else:
                        temp_chunk = []
                        temp_words = 0
                temp_chunk.append(sent)
                temp_words += sent_words
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
        
        # Normal paragraph processing
        elif current_words + para_words > target_words and current_chunk:
            # Finish current chunk
            chunks.append('\n\n'.join(current_chunk))
            # Start new chunk with overlap
            if len(current_chunk) > 1 and overlap > 0:
                overlap_para = current_chunk[-1]
                overlap_words = len(overlap_para.split())
                if overlap_words <= overlap:
                    current_chunk = [overlap_para, para]
                    current_words = overlap_words + para_words
                else:
                    current_chunk = [para]
                    current_words = para_words
            else:
                current_chunk = [para]
                current_words = para_words
        else:
            # Add to current chunk
            current_chunk.append(para)
            current_words += para_words
    
    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    # Filter out very short chunks (< 50 words)
    return [c for c in chunks if len(c.split()) >= 50]

def detect_language(text: str) -> str:
    """Simple language detection for German vs English content.
    
    Uses keyword frequency heuristics to avoid external dependencies.
    Returns 'de' for German, 'en' for English, 'unknown' for unclear.
    """
    if not text or len(text.split()) < 10:
        return 'unknown'
    
    text_lower = text.lower()
    
    # German indicators
    german_words = [
        'und', 'der', 'die', 'das', 'ist', 'zu', 'den', 'dem', 'ein', 'eine',
        'mit', 'für', 'von', 'auf', 'im', 'am', 'studium', 'universität',
        'hochschule', 'forschung', 'wissenschaft', 'studiengang', 'professor',
        'fakultät', 'bachelor', 'master', 'promotion', 'lehre', 'semester'
    ]
    
    # English indicators  
    english_words = [
        'the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by',
        'from', 'as', 'is', 'are', 'was', 'were', 'university', 'research',
        'study', 'program', 'professor', 'faculty', 'bachelor', 'master',
        'degree', 'course', 'semester', 'academic', 'education'
    ]
    
    german_count = sum(1 for word in german_words if word in text_lower)
    english_count = sum(1 for word in english_words if word in text_lower)
    
    if german_count > english_count and german_count >= 3:
        return 'de'
    elif english_count > german_count and english_count >= 3:
        return 'en'
    else:
        return 'unknown'
