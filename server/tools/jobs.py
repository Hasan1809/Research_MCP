"""MCP tools for local background jobs."""

from services.jobs.job_manager import (
    cancel_job,
    get_job_result,
    get_job_status,
    start_batch_build_profiles_job,
    start_batch_validate_gaps_job,
)
from config import WORKFLOW_MAX_PAPERS
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def start_batch_build_profiles_job_tool(
    papers: list[dict],
    force: bool = False,
    max_workers: int = 2,
    allow_large_batch: bool = False,
) -> dict:
    """
    Start a long-running batch profile build as a background job.

    This returns immediately with a job_id so the client can poll progress using
    get_job_status_tool instead of waiting for the full batch to finish.
    """
    arguments = {
        "paper_count": len(papers or []),
        "force": force,
        "max_workers": max_workers,
        "allow_large_batch": allow_large_batch,
    }
    logger.info(
        "Tool invoked: start_batch_build_profiles_job paper_count=%d force=%s max_workers=%d",
        len(papers or []),
        force,
        max_workers,
    )
    if len(papers or []) > WORKFLOW_MAX_PAPERS and not allow_large_batch:
        error = (
            f"start_batch_build_profiles_job received {len(papers or [])} papers. "
            f"The normal workflow cap is {WORKFLOW_MAX_PAPERS}; select the most relevant "
            "papers or pass allow_large_batch=True only when the user explicitly requested a larger corpus."
        )
        log_invocation("start_batch_build_profiles_job", arguments, error=error)
        raise ValueError(error)
    result = start_batch_build_profiles_job(
        papers=papers,
        force=force,
        max_workers=max_workers,
    )
    log_invocation("start_batch_build_profiles_job", arguments, output=result)
    return result


def start_batch_validate_gaps_job_tool(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
) -> dict:
    """
    Start long-running validation of all candidate research and methodological
    gaps for a project.

    The tool returns immediately with a job_id and saves progress/results to
    disk. Use get_job_status_tool and get_job_result_tool to retrieve progress
    and final validation summaries.
    """
    arguments = {
        "project": project,
        "max_results_per_gap": max_results_per_gap,
        "mode": mode,
        "max_workers": max_workers,
    }
    logger.info(
        "Tool invoked: start_batch_validate_gaps_job project=%r max_results_per_gap=%d mode=%s max_workers=%d",
        project,
        max_results_per_gap,
        mode,
        max_workers,
    )
    result = start_batch_validate_gaps_job(
        project=project,
        max_results_per_gap=max_results_per_gap,
        mode=mode,
        max_workers=max_workers,
    )
    log_invocation("start_batch_validate_gaps_job", arguments, output=result)
    return result


def get_job_status_tool(
    job_id: str,
    wait_seconds: int = 180,
    poll_interval_seconds: int = 5,
) -> dict:
    """
    Check progress of a background job started by a long-running tool.

    Returns completed, failed, pending counts, partial results, recent errors,
    active items, and current status. By default this waits up to 180 seconds
    before returning, so Claude Desktop does not spam quick status checks while
    long LLM/search calls are still in flight. Set wait_seconds=0 for an
    immediate snapshot.
    """
    result = get_job_status(
        job_id,
        wait_seconds=wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    log_invocation(
        "get_job_status_tool",
        {
            "job_id": job_id,
            "wait_seconds": wait_seconds,
            "poll_interval_seconds": poll_interval_seconds,
        },
        output={
            "job_id": result.get("job_id"),
            "status": result.get("status"),
            "completed": result.get("completed"),
            "failed": result.get("failed"),
            "pending": result.get("pending"),
            "waited_seconds": result.get("waited_seconds"),
        },
    )
    return result


def get_job_result_tool(job_id: str) -> dict:
    """
    Return the final result for a background job, including the result artifact
    path and errors. If the job is still running, returns the current status and
    any partial result metadata available.
    """
    result = get_job_result(job_id)
    log_invocation(
        "get_job_result_tool",
        {"job_id": job_id},
        output={
            "job_id": result.get("job_id"),
            "status": result.get("status"),
            "result_artifact_path": result.get("result_artifact_path"),
            "error_count": len(result.get("errors") or []),
        },
    )
    return result

def cancel_job_tool(job_id: str) -> dict:
    """
    Request cancellation of a running background job.

    Workers stop safely between items when possible; already-running external
    LLM/search calls may finish before the cancellation takes effect.
    """
    result = cancel_job(job_id)
    log_invocation(
        "cancel_job_tool",
        {"job_id": job_id},
        output={"job_id": result.get("job_id"), "status": result.get("status")},
    )
    return result
