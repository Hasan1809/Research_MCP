from services.documents.pdf_service import (
    download_and_extract_text, detect_sections_from_text,
    load_cached, save_cached,
)
from services.documents.chunking import chunk_text, chunk_sections
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def ingest_paper_tool(paper_id: str, source: str) -> dict:
    logger.info("Tool invoked: ingest_paper paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    if source != "arxiv":
        error = f"Unsupported source: {source}. Only 'arxiv' is supported."
        log_invocation("ingest_paper_tool", arguments, error=error)
        raise ValueError(error)

    cached = load_cached(source, paper_id)
    if cached is not None:
        logger.info("Returning cached result for paper_id=%r", paper_id)
        log_invocation("ingest_paper_tool", arguments, output={**{
            k: v for k, v in cached.items() if k != "chunks"
        }, "from_cache": True})
        return cached

    pdf_url = f"https://arxiv.org/pdf/{paper_id}"

    try:
        text = download_and_extract_text(pdf_url)
        chunks = chunk_text(text)
        logger.info("Flat chunking complete: %d chunks", len(chunks))

        # Structured extraction (section-aware) — falls back gracefully if no sections found
        structured = detect_sections_from_text(text)
        sections = structured.get("sections", [])
        sec_chunks = chunk_sections(sections) if sections else []
        logger.info("Structured extraction: %d sections, %d section chunks",
                    len(sections), len(sec_chunks))

        result = {
            "paper_id": paper_id,
            "source": source,
            "pdf_url": pdf_url,
            "text_length": len(text),
            "full_text": text,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "sections": sections,
            "section_chunks": sec_chunks,
            "metadata": structured.get("metadata", {}),
        }
        save_cached(source, paper_id, result)
        log_invocation("ingest_paper_tool", arguments, output={
            "paper_id": paper_id,
            "source": source,
            "pdf_url": pdf_url,
            "text_length": len(text),
            "chunk_count": len(chunks),
            "section_count": len(sections),
            "section_chunk_count": len(sec_chunks),
        })
        return result
    except Exception as e:
        log_invocation("ingest_paper_tool", arguments, error=str(e))
        raise
