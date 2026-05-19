from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BATCH_INGEST_MAX_WORKERS
from tools.ingest_paper import ingest_paper_tool
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def batch_ingest_papers_tool(papers: list[dict], max_workers: int = None) -> dict:
    """
    Download and process multiple paper PDFs concurrently.
    Ingest papers concurrently.

    papers must be a list of dicts with paper_id and source:
      [{"paper_id": "2602.07652", "source": "arxiv"}, ...]

    For source='arxiv', paper_id must be a bare arXiv ID like '2602.07652'.
    For source='semantic_scholar', paper_id must be the Semantic Scholar paperId
    returned by search_papers_tool. Never pass a URL as paper_id.

    Returns succeeded (list of paper_ids that completed successfully)
    and failed (dict of paper_id -> error string).

    Example:
      batch_ingest_papers_tool(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "649def34f8be52c8b66281af98ae884c09aef38b", "source": "semantic_scholar"}
      ])
    """
    arguments = {"papers": papers}
    succeeded = []
    failed = {}

    if not papers:
        log_invocation(
            "batch_ingest_papers_tool",
            arguments,
            output={
                "success_count": 0,
                "error_count": 0,
                "succeeded": succeeded,
                "failed": failed,
            },
        )
        return {"succeeded": succeeded, "failed": failed}

    if max_workers is None:
        max_workers = BATCH_INGEST_MAX_WORKERS
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    max_workers = min(len(papers), max_workers)
    logger.info("Batch ingest started: papers=%d max_workers=%d", len(papers), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                ingest_paper_tool,
                ref["paper_id"],
                ref["source"],
            ): ref["paper_id"]
            for ref in papers
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
                succeeded.append(paper_id)
                logger.info("Ingest complete: paper_id=%r", paper_id)
            except Exception as e:
                failed[paper_id] = str(e)
                logger.error(
                    "Ingest failed: paper_id=%r error=%s", paper_id, e
                )

    logger.info(
        "Batch ingest complete: success_count=%d failure_count=%d",
        len(succeeded),
        len(failed),
    )
    log_invocation(
        "batch_ingest_papers_tool",
        arguments,
        output={
            "success_count": len(succeeded),
            "error_count": len(failed),
            "succeeded": succeeded,
            "failed": failed,
        },
    )
    return {"succeeded": succeeded, "failed": failed}
