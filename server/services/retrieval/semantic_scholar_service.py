import os
import time
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 4
_BACKOFF_BASE = 5  # seconds; wait = base * 2^attempt


def fetch_papers(query: str, limit: int) -> list[dict]:
    api_key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    logger.info(
        "Semantic Scholar request: query=%r limit=%d key=%s",
        query, limit, "set" if api_key else "unset",
    )
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = httpx.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,year,authors,externalIds",
                },
                headers=headers,
                timeout=30,
            )
            if response.status_code == 429:
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Semantic Scholar rate-limited (429) — attempt %d/%d, retrying in %ds",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
        except httpx.HTTPError as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Semantic Scholar HTTP error on attempt %d/%d: %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                logger.exception("Semantic Scholar HTTP error (all retries exhausted): %s", e)
            continue

        papers = [
            {
                "paper_id": paper.get("externalIds", {}).get("ArXiv") or paper.get("paperId", ""),
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

    if last_exc:
        raise last_exc
    raise httpx.HTTPStatusError(
        "Semantic Scholar returned 429 on all retries",
        request=None,  # type: ignore[arg-type]
        response=None,  # type: ignore[arg-type]
    )
