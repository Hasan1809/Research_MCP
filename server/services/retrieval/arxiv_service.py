import httpx
import xml.etree.ElementTree as ET
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_papers(query: str, limit: int) -> list[dict]:
    logger.info("arXiv request: query=%r limit=%d", query, limit)
    try:
        response = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "max_results": limit},
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.exception("arXiv HTTP error: %s", e)
        raise
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    papers = [
        {
            "title": entry.findtext("atom:title", namespaces=ns).strip(),
            "abstract": entry.findtext("atom:summary", namespaces=ns).strip(),
            "year": entry.findtext("atom:published", namespaces=ns)[:4],
            "authors": [
                author.findtext("atom:name", namespaces=ns)
                for author in entry.findall("atom:author", namespaces=ns)
            ],
            "source": "arxiv",
        }
        for entry in root.findall("atom:entry", namespaces=ns)
    ]
    logger.info("arXiv returned %d papers", len(papers))
    return papers
