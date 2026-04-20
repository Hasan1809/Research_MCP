import json
import os
from services.retrieval.vector_store import index_chunks, index_structured_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "papers")


def index_paper_tool(paper_id: str, source: str) -> dict:
    """
    Index a paper's chunks into the vector database for semantic search.

    Must be called after ingest_paper_tool.
    Required before retrieve_chunks_tool can be used on this paper.
    Optional if you only need build_paper_profile_tool (profiling works
    without indexing for papers under 80k chars).
    """
    logger.info("Tool invoked: index_paper paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    cache_path = os.path.join(_DATA_DIR, source, f"{paper_id}.json")
    if not os.path.exists(cache_path):
        error = f"Paper not found in cache: {cache_path}. Run ingest_paper_tool first."
        log_invocation("index_paper_tool", arguments, error=error)
        raise FileNotFoundError(error)

    logger.info("Loading cached paper: %s", cache_path)
    with open(cache_path, encoding="utf-8") as f:
        paper = json.load(f)

    try:
        # Flat chunks (existing, backward-compatible)
        flat_chunks = paper.get("chunks", [])
        flat_counts = index_chunks(paper_id, source, flat_chunks)

        # Structured chunks (new — only present after the refactored ingest)
        sec_chunks = paper.get("section_chunks")
        structured_counts = {}
        if sec_chunks:
            logger.info("Indexing %d structured chunks", len(sec_chunks))
            structured_counts = index_structured_chunks(paper_id, source, sec_chunks)
        else:
            logger.info("No section_chunks in cache — skipping structured indexing")

        result = {
            "paper_id": paper_id,
            "source": source,
            "chunk_count": len(flat_chunks),
            "indexed_content_count": flat_counts["content_count"],
            "indexed_reference_count": flat_counts["reference_count"],
            "structured_chunk_count": len(sec_chunks) if sec_chunks else 0,
            **structured_counts,
        }
        log_invocation("index_paper_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("index_paper_tool", arguments, error=str(e))
        raise
