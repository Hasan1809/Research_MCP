from concurrent.futures import ThreadPoolExecutor, as_completed

from services.retrieval import arxiv_service, semantic_scholar_service
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_papers(query: str, limit: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(arxiv_service.fetch_papers, query, limit): "arxiv",
            pool.submit(semantic_scholar_service.fetch_papers, query, limit): "s2",
        }
        results = []
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                logger.warning("Search source failed: %s", e)
    return _deduplicate(results)


def _deduplicate(papers: list[dict]) -> list[dict]:
    seen = {}
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        if not paper_id:
            continue
        dedupe_key = paper_id
        if paper.get("source") == "semantic_scholar" and paper.get("arxiv_id"):
            dedupe_key = paper["arxiv_id"]
        if dedupe_key not in seen or paper.get("source") == "arxiv":
            seen[dedupe_key] = paper
    return list(seen.values())
