from services.documents.pdf_service import (
    download_and_extract_text, detect_sections_from_text,
    load_cached, save_cached,
)
from services.documents.chunking import chunk_text, chunk_sections
from services.citations import save_normalized_metadata
from services.retrieval.semantic_scholar_service import resolve_pdf_url
from config import DATA_DIR
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def _cache_path(source: str, paper_id: str) -> str:
    return str(DATA_DIR / "papers" / source / f"{paper_id}.json")


def _log_ingest_summary(
    source: str,
    paper_id: str,
    text_length: int,
    page_count: int,
    section_count: int,
    chunk_count: int,
    section_chunk_count: int,
) -> None:
    logger.info(
        "Ingest summary: source=%s paper_id=%s text_chars=%d pages=%d sections=%d "
        "flat_chunks=%d section_chunks=%d cache_path=%s",
        source,
        paper_id,
        text_length,
        page_count,
        section_count,
        chunk_count,
        section_chunk_count,
        _cache_path(source, paper_id),
    )


def ingest_paper_tool(paper_id: str, source: str) -> dict:
    """
    Download and process a paper PDF.

    For source='arxiv', paper_id must be a bare arXiv ID like '2602.07652'.
    For source='semantic_scholar', paper_id must be the Semantic Scholar paperId
    returned by search_papers_tool.
    Never pass a URL, filename, or placeholder as paper_id.

    Example: ingest_paper_tool(paper_id='2602.07652', source='arxiv')
    Example: ingest_paper_tool(paper_id='649def34f8be52c8b66281af98ae884c09aef38b', source='semantic_scholar')

    Caches the result so subsequent calls with the same paper_id are instant.
    Used by batch_ingest_papers_tool before profile-building jobs.
    """
    logger.info("Tool invoked: ingest_paper paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    if source not in {"arxiv", "semantic_scholar"}:
        error = f"Unsupported source: {source}. Use 'arxiv' or 'semantic_scholar'."
        log_invocation("ingest_paper_tool", arguments, error=error)
        raise ValueError(error)

    cached = load_cached(source, paper_id)
    if cached is not None:
        metadata = cached.get("metadata", {})
        _log_ingest_summary(
            source,
            paper_id,
            cached.get("text_length", 0),
            metadata.get("page_count", 0),
            len(cached.get("sections", [])),
            cached.get("chunk_count", len(cached.get("chunks", []))),
            len(cached.get("section_chunks", [])),
        )
        log_invocation("ingest_paper_tool", arguments, output={
            "paper_id": paper_id,
            "source": source,
            "pdf_url": cached.get("pdf_url", ""),
            "text_length": cached.get("text_length", 0),
            "chunk_count": cached.get("chunk_count", len(cached.get("chunks", []))),
            "section_count": len(cached.get("sections", [])),
            "section_chunk_count": len(cached.get("section_chunks", [])),
            "from_cache": True,
        })
        return cached

    metadata = {}
    if source == "arxiv":
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    else:
        pdf_url, metadata = resolve_pdf_url(paper_id)

    try:
        text = download_and_extract_text(pdf_url)
        chunks = chunk_text(text)

        # Structured extraction (section-aware) — falls back gracefully if no sections found
        structured = detect_sections_from_text(text)
        sections = structured.get("sections", [])
        sec_chunks = chunk_sections(sections) if sections else []
        structured_metadata = structured.get("metadata", {})

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
            "metadata": structured_metadata,
        }
        if metadata:
            result["semantic_scholar"] = metadata
        save_cached(source, paper_id, result)
        save_normalized_metadata(source, paper_id, {**metadata, "pdf_url": pdf_url})
        _log_ingest_summary(
            source,
            paper_id,
            len(text),
            structured_metadata.get("page_count", 0),
            len(sections),
            len(chunks),
            len(sec_chunks),
        )
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
