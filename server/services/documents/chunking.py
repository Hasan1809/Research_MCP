import re
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 100
OVERLAP_CHARS = 80


def chunk_text(text: str) -> list[str]:
    # Split on blank lines (paragraph boundaries), then merge or split as needed
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph exceeds the max, split it on sentence boundaries
        if len(para) > MAX_CHUNK_CHARS:
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                if len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS and len(current) >= MIN_CHUNK_CHARS:
                    chunks.append(current.strip())
                    current = current[-OVERLAP_CHARS:].lstrip() + " " + sentence
                else:
                    current = (current + " " + sentence).strip()
            continue

        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and len(current) >= MIN_CHUNK_CHARS:
            chunks.append(current.strip())
            current = current[-OVERLAP_CHARS:].lstrip() + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    avg_len = int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0
    logger.info(
        "Paragraph-aware chunking: chunks=%d avg_len=%d chars",
        len(chunks), avg_len,
    )
    return chunks
