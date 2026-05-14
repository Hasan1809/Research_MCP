import time
from typing import Any

import httpx

from config import S2_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 4
_BACKOFF_BASE = 5  # seconds; wait = base * 2^attempt
_PAPER_FIELDS = "title,abstract,year,venue,authors,externalIds,openAccessPdf,url"


def _headers() -> dict[str, str]:
    return {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


def _get_with_retries(url: str, *, params: dict[str, Any]) -> httpx.Response:
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = httpx.get(
                url,
                params=params,
                headers=_headers(),
                timeout=30,
            )
            if response.status_code == 429:
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Semantic Scholar rate-limited (429) - attempt %d/%d, retrying in %ds",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Semantic Scholar HTTP error on attempt %d/%d: %s - retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                logger.exception("Semantic Scholar HTTP error (all retries exhausted): %s", e)

    if last_exc:
        raise last_exc
    raise httpx.HTTPStatusError(
        "Semantic Scholar returned 429 on all retries",
        request=None,  # type: ignore[arg-type]
        response=None,  # type: ignore[arg-type]
    )


def _normalize_paper(paper: dict) -> dict:
    external_ids = paper.get("externalIds") or {}
    open_access_pdf = paper.get("openAccessPdf") or {}
    return {
        "paper_id": paper.get("paperId", ""),
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "year": str(paper["year"]) if paper.get("year") else "",
        "authors": [a.get("name", "") for a in paper.get("authors", [])],
        "venue": paper.get("venue") or "",
        "doi": external_ids.get("DOI", ""),
        "source": "semantic_scholar",
        "semantic_scholar_id": paper.get("paperId", ""),
        "external_ids": external_ids,
        "arxiv_id": external_ids.get("ArXiv", ""),
        "pdf_url": open_access_pdf.get("url") or "",
        "semantic_scholar_url": paper.get("url") or "",
    }


def fetch_papers(query: str, limit: int) -> list[dict]:
    logger.info(
        "Semantic Scholar request: query=%r limit=%d key=%s",
        query, limit, "set" if S2_API_KEY else "unset",
    )
    response = _get_with_retries(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": _PAPER_FIELDS,
        },
    )
    papers = [_normalize_paper(paper) for paper in response.json().get("data", [])]
    logger.info("Semantic Scholar returned %d papers", len(papers))
    return papers


def fetch_paper_details(paper_id: str) -> dict:
    logger.info("Semantic Scholar details request: paper_id=%r", paper_id)
    response = _get_with_retries(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
        params={"fields": _PAPER_FIELDS},
    )
    return _normalize_paper(response.json())


def resolve_pdf_url(paper_id: str) -> tuple[str, dict]:
    details = fetch_paper_details(paper_id)
    if details.get("pdf_url"):
        return details["pdf_url"], details
    if details.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{details['arxiv_id']}", details
    raise ValueError(
        f"No open-access PDF URL found for Semantic Scholar paper_id={paper_id!r}."
    )
