import json
import os
from services.retrieval.vector_store import index_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "papers")


def index_paper_tool(paper_id: str, source: str) -> dict:
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

    chunks = paper.get("chunks", [])

    try:
        counts = index_chunks(paper_id, source, chunks)
        result = {
            "paper_id": paper_id,
            "source": source,
            "chunk_count": len(chunks),
            "indexed_content_count": counts["content_count"],
            "indexed_reference_count": counts["reference_count"],
        }
        log_invocation("index_paper_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("index_paper_tool", arguments, error=str(e))
        raise
