import sys
import types
from pathlib import Path

from langchain_core.tools import tool

ROOT_DIR = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT_DIR / "server"
SERVER_TOOLS_DIR = SERVER_DIR / "tools"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

tools_pkg = types.ModuleType("tools")
tools_pkg.__path__ = [str(SERVER_TOOLS_DIR)]
sys.modules["tools"] = tools_pkg

from services.jobs.job_manager import (
    cancel_job as cancel_job_impl,
    get_job_result as get_job_result_impl,
    get_job_status as get_job_status_impl,
    start_batch_build_profiles_job as start_batch_build_profiles_job_impl,
    start_batch_validate_gaps_job as start_batch_validate_gaps_job_impl,
)
from config import WORKFLOW_MAX_PAPERS
from langchain_baseline.utils.logger import log_invocation


def _coerce_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return default
        return int(text)
    return int(value)


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


@tool
def start_batch_build_profiles_job(
    papers: list[dict] | str,
    force: bool | str = False,
    max_workers: int | str | None = 2,
    allow_large_batch: bool | str = False,
) -> dict:
    """Start batch profile building as a background job and return a job_id immediately."""
    if isinstance(papers, str):
        import json
        papers = json.loads(papers)
    allow_large = _coerce_bool(allow_large_batch)
    worker_count = _coerce_int(max_workers, 2)
    arguments = {
        "paper_count": len(papers or []),
        "force": force,
        "max_workers": worker_count,
        "allow_large_batch": allow_large,
    }
    if len(papers or []) > WORKFLOW_MAX_PAPERS and not allow_large:
        error = (
            f"start_batch_build_profiles_job received {len(papers or [])} papers. "
            f"The normal workflow cap is {WORKFLOW_MAX_PAPERS}; select the most relevant "
            "papers or pass allow_large_batch=True only when the user explicitly requested a larger corpus."
        )
        log_invocation("lc_start_batch_build_profiles_job", arguments, error=error)
        raise ValueError(error)
    result = start_batch_build_profiles_job_impl(
        papers=papers,
        force=_coerce_bool(force),
        max_workers=worker_count,
    )
    log_invocation("lc_start_batch_build_profiles_job", arguments, output=result)
    return result


@tool
def start_batch_validate_gaps_job(
    project: str,
    max_results_per_gap: int | str | None = 10,
    mode: str = "metadata_only",
    max_workers: int | str | None = 2,
) -> dict:
    """Start project gap validation as a background job and return a job_id immediately."""
    result = start_batch_validate_gaps_job_impl(
        project=project,
        max_results_per_gap=_coerce_int(max_results_per_gap, 10),
        mode=mode,
        max_workers=_coerce_int(max_workers, 2),
    )
    log_invocation(
        "lc_start_batch_validate_gaps_job",
        {
            "project": project,
            "max_results_per_gap": _coerce_int(max_results_per_gap, 10),
            "mode": mode,
            "max_workers": _coerce_int(max_workers, 2),
        },
        output=result,
    )
    return result


@tool
def get_job_status(
    job_id: str,
    wait_seconds: int | str | None = 180,
    poll_interval_seconds: int | str | None = 5,
) -> dict:
    """Check progress of a background job, waiting by default to avoid rapid polling."""
    wait = _coerce_int(wait_seconds, 180)
    poll = _coerce_int(poll_interval_seconds, 5)
    result = get_job_status_impl(
        job_id,
        wait_seconds=wait,
        poll_interval_seconds=poll,
    )
    log_invocation(
        "lc_get_job_status",
        {"job_id": job_id, "wait_seconds": wait, "poll_interval_seconds": poll},
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


@tool
def get_job_result(job_id: str) -> dict:
    """Fetch the final or current result for a background job."""
    result = get_job_result_impl(job_id)
    log_invocation(
        "lc_get_job_result",
        {"job_id": job_id},
        output={
            "job_id": result.get("job_id"),
            "status": result.get("status"),
            "result_artifact_path": result.get("result_artifact_path"),
            "error_count": len(result.get("errors") or []),
        },
    )
    return result

@tool
def cancel_job(job_id: str) -> dict:
    """Request cancellation of a background job."""
    result = cancel_job_impl(job_id)
    log_invocation(
        "lc_cancel_job",
        {"job_id": job_id},
        output={"job_id": result.get("job_id"), "status": result.get("status")},
    )
    return result
