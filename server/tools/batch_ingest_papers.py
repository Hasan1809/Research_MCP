from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BATCH_INGEST_MAX_WORKERS, WORKFLOW_MAX_PAPERS
from tools.ingest_paper import ingest_paper_tool
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def _normalize_source(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if normalized in {"semanticscholar", "semantic-scholar", "semantic scholar"}:
        return "semantic_scholar"
    return normalized


def batch_ingest_papers_tool(
    papers: list[dict],
    max_workers: int = None,
    allow_large_batch: bool = False,
) -> dict:
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
    normalized_papers = []
    for ref in papers or []:
        if isinstance(ref, dict):
            normalized_papers.append({
                **ref,
                "paper_id": str(ref.get("paper_id") or "").strip(),
                "source": _normalize_source(ref.get("source")),
            })
        else:
            normalized_papers.append(ref)

    arguments = {
        "papers": normalized_papers,
        "allow_large_batch": allow_large_batch,
    }
    succeeded = []
    succeeded_papers = []
    failed = {}

    if not normalized_papers:
        log_invocation(
            "batch_ingest_papers_tool",
            arguments,
            output={
                "success_count": 0,
                "error_count": 0,
                "succeeded": succeeded,
                "succeeded_papers": succeeded_papers,
                "failed": failed,
            },
        )
        return {"succeeded": succeeded, "succeeded_papers": succeeded_papers, "failed": failed}

    if len(normalized_papers) > WORKFLOW_MAX_PAPERS and not allow_large_batch:
        error = (
            f"batch_ingest_papers_tool received {len(normalized_papers)} papers. "
            f"The normal workflow cap is {WORKFLOW_MAX_PAPERS}; select the most relevant "
            "papers or pass allow_large_batch=True only when the user explicitly requested a larger corpus."
        )
        log_invocation("batch_ingest_papers_tool", arguments, error=error)
        raise ValueError(error)

    if max_workers is None:
        max_workers = BATCH_INGEST_MAX_WORKERS
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    max_workers = min(len(normalized_papers), max_workers)
    logger.info("Batch ingest started: papers=%d max_workers=%d", len(normalized_papers), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                ingest_paper_tool,
                ref["paper_id"],
                ref["source"],
            ): ref
            for ref in normalized_papers
        }
        for future in as_completed(futures):
            ref = futures[future]
            paper_id = ref.get("paper_id", "")
            try:
                future.result()
                succeeded.append(paper_id)
                succeeded_papers.append({
                    "paper_id": paper_id,
                    "source": ref.get("source", ""),
                })
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
            "succeeded_papers": succeeded_papers,
            "failed": failed,
        },
    )
    return {"succeeded": succeeded, "succeeded_papers": succeeded_papers, "failed": failed}
