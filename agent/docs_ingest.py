# small helpers for document ingestion (optional)
# Currently ingestion is implemented directly in app.py but this module
# can be extended to support more file formats, OCR, or chunking strategies.

from typing import List

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError('chunk_size must be > overlap')
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]
