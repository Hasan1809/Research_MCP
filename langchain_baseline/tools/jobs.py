from langchain_core.tools import tool

from services.jobs.job_manager import (
    cancel_job as cancel_job_impl,
    get_job_result as get_job_result_impl,
    get_job_status as get_job_status_impl,
    list_jobs as list_jobs_impl,
    start_batch_build_profiles_job as start_batch_build_profiles_job_impl,
    start_batch_validate_gaps_job as start_batch_validate_gaps_job_impl,
)


@tool
def start_batch_build_profiles_job(
    papers: list[dict] | str,
    force: bool = False,
    max_workers: int = 2,
) -> dict:
    """Start batch profile building as a background job and return a job_id immediately."""
    if isinstance(papers, str):
        import json
        papers = json.loads(papers)
    return start_batch_build_profiles_job_impl(
        papers=papers,
        force=force,
        max_workers=max_workers,
    )


@tool
def start_batch_validate_gaps_job(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
) -> dict:
    """Start project gap validation as a background job and return a job_id immediately."""
    return start_batch_validate_gaps_job_impl(
        project=project,
        max_results_per_gap=max_results_per_gap,
        mode=mode,
        max_workers=max_workers,
    )


@tool
def get_job_status(job_id: str) -> dict:
    """Check progress of a background job."""
    return get_job_status_impl(job_id)


@tool
def get_job_result(job_id: str) -> dict:
    """Fetch the final or current result for a background job."""
    return get_job_result_impl(job_id)


@tool
def list_jobs(status: str | None = None, limit: int = 20) -> dict:
    """List recent background jobs."""
    return list_jobs_impl(status=status, limit=limit)


@tool
def cancel_job(job_id: str) -> dict:
    """Request cancellation of a background job."""
    return cancel_job_impl(job_id)
