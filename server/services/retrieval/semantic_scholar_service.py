import os
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_papers(query: str, limit: int) -> list[dict]:
    api_key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    logger.info("Semantic Scholar request: query=%r limit=%d key=%s", query, limit, "set" if api_key else "unset")
    try:
        response = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,authors",
            },
            headers=headers,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.exception("Semantic Scholar HTTP error: %s", e)
        raise
    papers = [
        {
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "year": str(paper["year"]) if paper.get("year") else "",
            "authors": [a["name"] for a in paper.get("authors", [])],
            "source": "semantic_scholar",
        }
        for paper in response.json().get("data", [])
    ]
    logger.info("Semantic Scholar returned %d papers", len(papers))
    return papers
