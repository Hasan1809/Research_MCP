from services.retrieval.aggregator import fetch_papers
from services.citations import save_search_metadata
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def search_papers_tool(query: str, limit: int) -> list[dict]:
    """
    Search for academic papers across arXiv and Semantic Scholar.

    Returns papers with paper_id, title, abstract, year, authors, source.
    Use the paper_id and source from results to call ingest_paper_tool.
    Papers with source='arxiv' or source='semantic_scholar' can be ingested.
    """
    logger.info("Tool invoked: query=%r limit=%d", query, limit)
    try:
        result = fetch_papers(query, limit)
        save_search_metadata(result)
        log_invocation("search_papers_tool", {"query": query, "limit": limit}, output=result)
        return result
    except Exception as e:
        log_invocation("search_papers_tool", {"query": query, "limit": limit}, error=str(e))
        raise
