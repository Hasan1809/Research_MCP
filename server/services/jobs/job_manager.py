from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Callable

from config import DATA_DIR
from services.analysis import gap_validator
from tools.build_paper_profile import build_paper_profile_tool
from utils.logger import get_logger

logger = get_logger(__name__)

JOBS_DIR = DATA_DIR / "jobs"
JOB_RESULTS_DIR = JOBS_DIR / "results"
_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LOCK = RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_job_id(job_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{job_type}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _result_path(job_id: str) -> Path:
    return JOB_RESULTS_DIR / f"{job_id}_result.json"


def _read_job(job_id: str) -> dict:
    path = _job_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"No job found for job_id={job_id!r}.")
    last_exc = None
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError as e:
            last_exc = e
            time.sleep(0.05)
    if last_exc:
        raise last_exc
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(job: dict) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job["last_updated"] = _now()
    path = _job_path(job["job_id"])
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    last_exc = None
    for _ in range(5):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as e:
            last_exc = e
            time.sleep(0.05)
    if last_exc:
        raise last_exc


def _update_job(job_id: str, updater: Callable[[dict], None]) -> dict:
    with _LOCK:
        job = _read_job(job_id)
        updater(job)
        _write_job(job)
        return job


def create_job(job_type: str, inputs: dict, total: int = 0) -> dict:
    job_id = _new_job_id(job_type)
    job = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "pending",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "last_updated": _now(),
        "total": total,
        "completed": 0,
        "failed": 0,
        "pending": total,
        "current_item": None,
        "active_items": [],
        "inputs": inputs,
        "partial_results": [],
        "errors": [],
        "result": None,
        "result_artifact_path": None,
    }
    with _LOCK:
        _write_job(job)
    return job


def _compact_status(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "total": job.get("total", 0),
        "completed": job.get("completed", 0),
        "failed": job.get("failed", 0),
        "pending": job.get("pending", 0),
        "started_at": job.get("started_at"),
        "last_updated": job.get("last_updated"),
        "finished_at": job.get("finished_at"),
        "current_item": job.get("current_item"),
        "active_items": job.get("active_items", []),
        "partial_results": job.get("partial_results", [])[-20:],
        "errors": job.get("errors", [])[-20:],
        "result_artifact_path": job.get("result_artifact_path"),
        "message": _status_message(job),
        "recommended_next_poll_seconds": _recommended_next_poll_seconds(job),
    }


def _status_message(job: dict) -> str:
    status = job.get("status")
    if status == "completed":
        return "Job completed. Use get_job_result_tool for full results."
    if status == "failed":
        return "Job failed. Use get_job_result_tool for error details."
    if status == "cancel_requested":
        return "Cancellation requested. The worker will stop between items when possible."
    if status == "cancelled":
        return "Job cancelled. Use get_job_result_tool for partial results."
    if job.get("active_items"):
        return (
            "Job is running. Active items may be waiting on slow LLM or academic search calls; "
            "this is expected. Poll get_job_status_tool with wait_seconds=180 instead of "
            "falling back to individual tools."
        )
    return "Job is running. Poll get_job_status_tool with wait_seconds=180 for progress."


def _recommended_next_poll_seconds(job: dict) -> int:
    if job.get("status") in {"completed", "failed", "cancelled"}:
        return 0
    if job.get("job_type") == "batch_build_profiles":
        return 120
    return 30


def get_job_status(
    job_id: str,
    wait_seconds: int = 0,
    poll_interval_seconds: int = 5,
) -> dict:
    wait_seconds = max(0, min(int(wait_seconds or 0), 210))
    poll_interval_seconds = max(1, min(int(poll_interval_seconds or 5), 30))
    deadline = time.time() + wait_seconds
    waited = 0.0
    last_status = None
    try:
        while True:
            last_status = _compact_status(_read_job(job_id))
            if last_status["status"] in {"completed", "failed", "cancelled", "not_found"}:
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            sleep_for = min(poll_interval_seconds, remaining)
            time.sleep(sleep_for)
            waited += sleep_for
        last_status["waited_seconds"] = round(waited, 2)
        return last_status
    except FileNotFoundError as e:
        return {"job_id": job_id, "status": "not_found", "error": str(e)}


def get_job_result(job_id: str) -> dict:
    try:
        job = _read_job(job_id)
    except FileNotFoundError as e:
        return {"job_id": job_id, "status": "not_found", "error": str(e)}
    result = job.get("result")
    artifact_path = job.get("result_artifact_path")
    if result is None and artifact_path and Path(artifact_path).exists():
        result = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    return {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "result": result,
        "result_artifact_path": artifact_path,
        "errors": job.get("errors", []),
    }


def list_jobs(status: str | None = None, limit: int = 20) -> dict:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not read job file: %s", path)
            continue
        if status and job.get("status") != status:
            continue
        jobs.append(_compact_status(job))
        if len(jobs) >= limit:
            break
    return {"jobs": jobs, "count": len(jobs)}


def cancel_job(job_id: str) -> dict:
    try:
        return _compact_status(_update_job(job_id, lambda job: job.update({"status": "cancel_requested"})))
    except FileNotFoundError as e:
        return {"job_id": job_id, "status": "not_found", "error": str(e)}


def _is_cancel_requested(job_id: str) -> bool:
    try:
        return _read_job(job_id).get("status") == "cancel_requested"
    except FileNotFoundError:
        return True


def _save_result(job_id: str, result: dict) -> str:
    JOB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _result_path(job_id)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)


def start_batch_build_profiles_job(
    papers: list[dict],
    force: bool = False,
    max_workers: int = 2,
) -> dict:
    if not papers:
        raise ValueError("papers must contain at least one paper reference.")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    inputs = {"papers": papers, "force": force, "max_workers": max_workers}
    job = create_job("batch_build_profiles", inputs, total=len(papers))
    _EXECUTOR.submit(_run_batch_build_profiles_job, job["job_id"])
    return {
        "status": "started",
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "paper_count": len(papers),
        "message": (
            "Batch profile build started. Use get_job_status_tool(job_id, wait_seconds=180) "
            "to monitor progress; do not fall back to individual profile calls while the job is running."
        ),
    }


def _profile_path(source: str, paper_id: str) -> str:
    return str(DATA_DIR / "profiles" / source / f"{paper_id}.json")


def _run_batch_build_profiles_job(job_id: str) -> None:
    started = time.time()
    job = _read_job(job_id)
    papers = job["inputs"]["papers"]
    force = job["inputs"].get("force", False)
    max_workers = min(int(job["inputs"].get("max_workers") or 2), len(papers))
    profiles = {}
    failed = {}

    def mark_running(job: dict) -> None:
        job["status"] = "running"
        job["started_at"] = job["started_at"] or _now()
        job["pending"] = job["total"]
        job["active_items"] = []

    _update_job(job_id, mark_running)

    def run_one(ref: dict) -> dict:
        if _is_cancel_requested(job_id):
            raise RuntimeError("cancel_requested")
        paper_id = ref["paper_id"]
        source = ref["source"]

        def mark_active(job: dict) -> None:
            active = job.setdefault("active_items", [])
            item = {"paper_id": paper_id, "source": source}
            if item not in active:
                active.append(item)
            job["current_item"] = paper_id

        _update_job(job_id, mark_active)
        profile = build_paper_profile_tool(paper_id, source, force)
        return {
            "paper_id": paper_id,
            "source": source,
            "status": "completed",
            "profile_path": _profile_path(source, paper_id),
            "profile": profile,
        }

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one, ref): ref for ref in papers}
            for future in as_completed(futures):
                ref = futures[future]
                paper_id = ref.get("paper_id", "")
                source = ref.get("source", "")
                if _is_cancel_requested(job_id):
                    for pending in futures:
                        pending.cancel()
                try:
                    item = future.result()
                    profiles[paper_id] = item["profile"]
                    partial = {
                        "paper_id": paper_id,
                        "source": source,
                        "status": "completed",
                        "profile_path": item["profile_path"],
                    }

                    def on_success(job: dict) -> None:
                        job["completed"] += 1
                        job["pending"] = max(job["total"] - job["completed"] - job["failed"], 0)
                        job["current_item"] = paper_id
                        job["active_items"] = [
                            item for item in job.get("active_items", [])
                            if item.get("paper_id") != paper_id or item.get("source") != source
                        ]
                        job["partial_results"].append(partial)

                    _update_job(job_id, on_success)
                except Exception as e:
                    error = "cancel_requested" if str(e) == "cancel_requested" else str(e)
                    failed[paper_id] = error

                    def on_failure(job: dict) -> None:
                        job["failed"] += 1
                        job["pending"] = max(job["total"] - job["completed"] - job["failed"], 0)
                        job["current_item"] = paper_id
                        job["active_items"] = [
                            item for item in job.get("active_items", [])
                            if item.get("paper_id") != paper_id or item.get("source") != source
                        ]
                        job["errors"].append({
                            "paper_id": paper_id,
                            "source": source,
                            "error": error,
                        })

                    _update_job(job_id, on_failure)
        result = {
            "success_count": len(profiles),
            "failure_count": len(failed),
            "profiles": profiles,
            "failed": failed,
            "profile_paths": {
                paper_id: _profile_path(profile.get("source", ""), paper_id)
                for paper_id, profile in profiles.items()
            },
            "duration_seconds": round(time.time() - started, 2),
        }
        result_path = _save_result(job_id, result)

        def finish(job: dict) -> None:
            job["status"] = "cancelled" if _is_cancel_requested(job_id) else "completed"
            job["finished_at"] = _now()
            job["current_item"] = None
            job["active_items"] = []
            job["pending"] = 0
            job["result"] = {
                "success_count": result["success_count"],
                "failure_count": result["failure_count"],
                "profile_paths": result["profile_paths"],
                "duration_seconds": result["duration_seconds"],
            }
            job["result_artifact_path"] = result_path

        _update_job(job_id, finish)
    except Exception as e:
        logger.exception("Batch profile job failed: job_id=%s", job_id)

        def fail(job: dict) -> None:
            job["status"] = "failed"
            job["finished_at"] = _now()
            job["active_items"] = []
            job["errors"].append({"error": str(e)})

        _update_job(job_id, fail)


def start_batch_validate_gaps_job(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
) -> dict:
    if not project:
        raise ValueError("project is required.")
    cached = gap_validator.find_existing_gap_analysis_for_project(project)
    if cached is None:
        raise FileNotFoundError(
            f"No cached gap analysis found for project {project!r}. Run detect_gaps_tool(project=...) first."
        )
    gap_entries = gap_validator._gap_entries(cached["analysis"])
    inputs = {
        "project": project,
        "max_results_per_gap": max_results_per_gap,
        "mode": mode,
        "max_workers": max_workers,
    }
    job = create_job("batch_validate_gaps", inputs, total=len(gap_entries))
    _EXECUTOR.submit(_run_batch_validate_gaps_job, job["job_id"])
    return {
        "status": "started",
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "project": project,
        "gap_count": len(gap_entries),
        "message": (
            "Batch gap validation started. Use get_job_status_tool(job_id, wait_seconds=180) "
            "to monitor progress."
        ),
    }


def _run_batch_validate_gaps_job(job_id: str) -> None:
    job = _read_job(job_id)
    inputs = job["inputs"]

    def mark_running(state: dict) -> None:
        state["status"] = "running"
        state["started_at"] = state["started_at"] or _now()
        state["active_items"] = []

    _update_job(job_id, mark_running)

    def progress(status: str, entry: dict, result: dict | None, error: str | None) -> None:
        def update(state: dict) -> None:
            state["current_item"] = entry.get("gap_id")
            if status == "running":
                active = state.setdefault("active_items", [])
                item = {
                    "gap_id": entry.get("gap_id"),
                    "gap_type": entry.get("gap_type"),
                    "gap": entry.get("gap"),
                }
                if not any(existing.get("gap_id") == entry.get("gap_id") for existing in active):
                    active.append(item)
                return

            state["active_items"] = [
                item for item in state.get("active_items", [])
                if item.get("gap_id") != entry.get("gap_id")
            ]
            if status == "completed" and result:
                state["completed"] += 1
                state["partial_results"].append({
                    "gap_id": result.get("gap_id"),
                    "gap_type": result.get("gap_type"),
                    "original_gap": result.get("original_gap"),
                    "status": result.get("status"),
                    "confidence": result.get("confidence"),
                    "use_for_experiments": result.get("use_for_experiments"),
                    "refined_gap": result.get("refined_gap"),
                    "artifact_path": result.get("artifact_path"),
                })
            else:
                state["failed"] += 1
                state["errors"].append({
                    "gap_id": entry.get("gap_id"),
                    "gap_type": entry.get("gap_type"),
                    "original_gap": entry.get("gap"),
                    "error": error or status,
                })
            state["pending"] = max(state["total"] - state["completed"] - state["failed"], 0)

        _update_job(job_id, update)

    try:
        result = gap_validator.batch_validate_gaps(
            project=inputs["project"],
            max_results_per_gap=inputs.get("max_results_per_gap", 10),
            mode=inputs.get("mode", "metadata_only"),
            max_workers=inputs.get("max_workers", 2),
            progress_callback=progress,
            cancel_check=lambda: _is_cancel_requested(job_id),
        )
        result_path = _save_result(job_id, result)

        def finish(state: dict) -> None:
            state["status"] = "cancelled" if _is_cancel_requested(job_id) else "completed"
            state["finished_at"] = _now()
            state["current_item"] = None
            state["active_items"] = []
            state["pending"] = 0
            state["result"] = {
                "project": result.get("project"),
                "gap_count": result.get("gap_count"),
                "validated_count": result.get("validated_count"),
                "failed_count": result.get("failed_count"),
                "status_counts": result.get("status_counts"),
                "batch_artifact_path": result.get("batch_artifact_path"),
            }
            state["result_artifact_path"] = result_path

        _update_job(job_id, finish)
    except Exception as e:
        logger.exception("Batch validation job failed: job_id=%s", job_id)

        def fail(state: dict) -> None:
            state["status"] = "failed"
            state["finished_at"] = _now()
            state["active_items"] = []
            state["errors"].append({"error": str(e)})

        _update_job(job_id, fail)
