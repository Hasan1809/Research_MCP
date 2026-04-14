import re
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 100
OVERLAP_CHARS = 80

_CONCLUSION_KEYWORDS = {"conclusion", "conclusions", "summary", "discussion"}
_ABSTRACT_KEYWORDS = {"abstract"}


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


def _chunk_section_text(text: str) -> list[str]:
    """Apply paragraph-aware chunking to a single section's text."""
    return chunk_text(text) if text.strip() else []


def chunk_sections(sections: list[dict]) -> list[dict]:
    """
    Chunk text respecting section boundaries.

    Each returned chunk dict contains:
        text, section, section_index, chunk_in_section,
        is_abstract, is_conclusion, position_ratio
    """
    total_sections = len(sections)
    result: list[dict] = []

    for sec_idx, section in enumerate(sections):
        heading_lower = section.get("heading", "").lower()
        is_abstract = any(k in heading_lower for k in _ABSTRACT_KEYWORDS)
        is_conclusion = any(k in heading_lower for k in _CONCLUSION_KEYWORDS)

        text_chunks = _chunk_section_text(section.get("text", ""))

        # Short sections with no chunks still get one entry if they have text
        if not text_chunks and section.get("text", "").strip():
            text_chunks = [section["text"].strip()]

        for chunk_idx, chunk_text_val in enumerate(text_chunks):
            # position_ratio: where this chunk sits in the overall document
            pos = (sec_idx + (chunk_idx / max(len(text_chunks), 1))) / max(total_sections, 1)
            result.append({
                "text": chunk_text_val,
                "section": section.get("heading", ""),
                "section_index": sec_idx,
                "chunk_in_section": chunk_idx,
                "is_abstract": is_abstract,
                "is_conclusion": is_conclusion,
                "position_ratio": round(pos, 4),
            })

    avg_len = int(sum(len(c["text"]) for c in result) / len(result)) if result else 0
    logger.info(
        "Section-aware chunking: %d sections → %d chunks, avg_len=%d",
        total_sections, len(result), avg_len,
    )
    return result
