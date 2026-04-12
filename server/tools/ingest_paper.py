from services.documents.pdf_service import download_and_extract_text
from services.documents.chunking import chunk_text
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def ingest_paper_tool(paper_id: str, source: str) -> dict:
    logger.info("Tool invoked: ingest_paper paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    if source != "arxiv":
        error = f"Unsupported source: {source}. Only 'arxiv' is supported."
        log_invocation("ingest_paper_tool", arguments, error=error)
        raise ValueError(error)

    pdf_url = f"https://arxiv.org/pdf/{paper_id}"

    try:
        text = download_and_extract_text(pdf_url)
        chunks = chunk_text(text)
        logger.info("Chunking complete: %d chunks", len(chunks))

        result = {
            "paper_id": paper_id,
            "source": source,
            "pdf_url": pdf_url,
            "text_length": len(text),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        log_invocation("ingest_paper_tool", arguments, output={
            "paper_id": paper_id,
            "source": source,
            "pdf_url": pdf_url,
            "text_length": len(text),
            "chunk_count": len(chunks),
        })
        return result
    except Exception as e:
        log_invocation("ingest_paper_tool", arguments, error=str(e))
        raise
