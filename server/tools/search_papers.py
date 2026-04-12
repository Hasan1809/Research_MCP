from services.retrieval.aggregator import fetch_papers
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def search_papers_tool(query: str, limit: int) -> list[dict]:
    logger.info("Tool invoked: query=%r limit=%d", query, limit)
    try:
        result = fetch_papers(query, limit)
        log_invocation("search_papers_tool", {"query": query, "limit": limit}, output=result)
        return result
    except Exception as e:
        log_invocation("search_papers_tool", {"query": query, "limit": limit}, error=str(e))
        raise
