from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BATCH_BUILD_PROFILES_MAX_WORKERS
from tools.build_paper_profile import build_paper_profile_tool
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def batch_build_profiles_tool(
    papers: list[dict],
    force: bool = False,
    max_workers: int = None,
) -> dict:
    """
    Build profiles for multiple papers concurrently using parallel
    IONOS LLM calls. Each call is independent with its own context -
    running them in parallel does not affect profile quality.
    Faster than calling build_paper_profile_tool one at a time.

    papers must be a list of dicts with paper_id and source:
      [{"paper_id": "2602.07652", "source": "arxiv"}, ...]

    For source='arxiv', paper_id must be a bare arXiv ID like '2602.07652'.
    For source='semantic_scholar', paper_id must be the Semantic Scholar paperId
    returned by search_papers_tool. Never pass a URL as paper_id.

    Each paper must have been ingested with batch_ingest_papers_tool
    or ingest_paper_tool first.

    Returns profiles (dict of paper_id -> profile) for successes and
    failed (dict of paper_id -> error string) for failures.

    max_workers controls profile calls run in parallel. The default comes from
    BATCH_BUILD_PROFILES_MAX_WORKERS. For larger Claude Desktop workflows,
    prefer start_batch_build_profiles_job so the client can poll progress
    instead of waiting for one long tool response.

    Example:
      batch_build_profiles_tool(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "649def34f8be52c8b66281af98ae884c09aef38b", "source": "semantic_scholar"}
      ])
    """
    arguments = {"papers": papers, "force": force, "max_workers": max_workers}
    profiles = {}
    failed = {}

    if not papers:
        log_invocation(
            "batch_build_profiles_tool",
            arguments,
            output={
                "success_count": 0,
                "error_count": 0,
                "succeeded": [],
                "failed": [],
            },
        )
        return {"profiles": profiles, "failed": failed}

    if max_workers is None:
        max_workers = BATCH_BUILD_PROFILES_MAX_WORKERS
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    max_workers = min(len(papers), max_workers)
    logger.info(
        "Batch profile build started: papers=%d max_workers=%d force=%s",
        len(papers),
        max_workers,
        force,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                build_paper_profile_tool,
                ref["paper_id"],
                ref["source"],
                force,
            ): ref["paper_id"]
            for ref in papers
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                profiles[paper_id] = future.result()
                logger.info("Profile complete: paper_id=%r", paper_id)
            except Exception as e:
                failed[paper_id] = str(e)
                logger.error(
                    "Profile failed: paper_id=%r error=%s", paper_id, e
                )

    logger.info(
        "Batch profile build complete: success_count=%d failure_count=%d",
        len(profiles),
        len(failed),
    )
    log_invocation(
        "batch_build_profiles_tool",
        arguments,
        output={
            "success_count": len(profiles),
            "error_count": len(failed),
            "succeeded": list(profiles.keys()),
            "failed": list(failed.keys()),
        },
    )
    return {"profiles": profiles, "failed": failed}
