import time
import httpx
import xml.etree.ElementTree as ET
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 4
_BACKOFF_BASE = 5  # seconds; wait = base * 2^attempt


def fetch_papers(query: str, limit: int) -> list[dict]:
    logger.info("arXiv request: query=%r limit=%d", query, limit)
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = httpx.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": query, "max_results": limit},
                timeout=30,
            )
            if response.status_code == 429:
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "arXiv rate-limited (429) — attempt %d/%d, retrying in %ds",
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
                    "arXiv HTTP error on attempt %d/%d: %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                logger.exception("arXiv HTTP error (all retries exhausted): %s", e)
            continue

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.text)
        papers = [
            {
                "paper_id": entry.findtext("atom:id", namespaces=ns).split("/abs/")[-1].split("v")[0].strip(),
                "title": entry.findtext("atom:title", namespaces=ns).strip(),
                "abstract": entry.findtext("atom:summary", namespaces=ns).strip(),
                "year": entry.findtext("atom:published", namespaces=ns)[:4],
                "authors": [
                    author.findtext("atom:name", namespaces=ns)
                    for author in entry.findall("atom:author", namespaces=ns)
                ],
                "venue": "",
                "doi": "",
                "arxiv_id": entry.findtext("atom:id", namespaces=ns).split("/abs/")[-1].split("v")[0].strip(),
                "semantic_scholar_id": "",
                "url": entry.findtext("atom:id", namespaces=ns).strip(),
                "pdf_url": f"https://arxiv.org/pdf/{entry.findtext('atom:id', namespaces=ns).split('/abs/')[-1].split('v')[0].strip()}",
                "source": "arxiv",
            }
            for entry in root.findall("atom:entry", namespaces=ns)
        ]
        logger.info("arXiv returned %d papers", len(papers))
        return papers

    if last_exc:
        raise last_exc
    raise httpx.HTTPStatusError(
        "arXiv returned 429 on all retries",
        request=None,  # type: ignore[arg-type]
        response=None,  # type: ignore[arg-type]
    )
