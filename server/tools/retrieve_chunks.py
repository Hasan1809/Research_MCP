from services.retrieval.vector_store import query_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def retrieve_chunks_tool(query: str, paper_id: str, source: str, k: int) -> list[dict]:
    logger.info("Tool invoked: retrieve_chunks query=%r paper_id=%r source=%r k=%d", query, paper_id, source, k)
    arguments = {"query": query, "paper_id": paper_id, "source": source, "k": k}
    try:
        result = query_chunks(query, paper_id, source, k)
        log_invocation("retrieve_chunks_tool", arguments, output={
            "chunk_count": len(result),
            "chunk_indices": [c["chunk_index"] for c in result],
        })
        return result
    except Exception as e:
        log_invocation("retrieve_chunks_tool", arguments, error=str(e))
        raise
