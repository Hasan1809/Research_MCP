import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.jobs import job_manager
from tools import jobs as jobs_tool_module


def _wait_for(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = job_manager.get_job_status(job_id)
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {job_manager.get_job_status(job_id)}")


def test_job_store_create_update_complete_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(job_manager, "JOB_RESULTS_DIR", tmp_path / "jobs" / "results")

    job = job_manager.create_job("test_job", {"x": 1}, total=2)
    job_path = tmp_path / "jobs" / f"{job['job_id']}.json"
    assert job_path.exists()

    job_manager._update_job(
        job["job_id"],
        lambda state: state.update({
            "status": "running",
            "completed": 1,
            "failed": 0,
            "pending": 1,
            "current_item": "item-1",
        }),
    )
    status = job_manager.get_job_status(job["job_id"])
    assert status["completed"] == 1
    assert status["current_item"] == "item-1"

    result_path = job_manager._save_result(job["job_id"], {"ok": True})
    job_manager._update_job(
        job["job_id"],
        lambda state: state.update({
            "status": "completed",
            "result": {"ok": True},
            "result_artifact_path": result_path,
        }),
    )
    result = job_manager.get_job_result(job["job_id"])
    assert result["status"] == "completed"
    assert result["result"] == {"ok": True}
    assert job_manager.list_jobs()["count"] == 1


def test_missing_job_returns_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path / "jobs")
    assert job_manager.get_job_status("missing")["status"] == "not_found"
    assert job_manager.get_job_result("missing")["status"] == "not_found"


def test_batch_profile_job_updates_after_each_paper_and_continues_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(job_manager, "JOB_RESULTS_DIR", tmp_path / "jobs" / "results")
    monkeypatch.setattr(job_manager, "DATA_DIR", tmp_path / "data")

    def fake_build(paper_id, source, force=False):
        if paper_id == "bad":
            raise RuntimeError("profile failed")
        path = tmp_path / "data" / "profiles" / source / f"{paper_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"paper_id": paper_id, "source": source, "title": f"Title {paper_id}"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(job_manager, "build_paper_profile_tool", fake_build)

    started = job_manager.start_batch_build_profiles_job(
        papers=[
            {"paper_id": "good", "source": "arxiv"},
            {"paper_id": "bad", "source": "arxiv"},
        ],
        max_workers=2,
    )

    assert started["status"] == "started"
    status = _wait_for(started["job_id"])
    assert status["status"] == "completed"
    assert status["completed"] == 1
    assert status["failed"] == 1
    assert status["pending"] == 0

    result = job_manager.get_job_result(started["job_id"])
    assert result["result"]["success_count"] == 1
    assert result["result"]["failure_count"] == 1
    assert os.path.exists(result["result"]["profile_paths"]["good"])
    assert result["errors"][0]["paper_id"] == "bad"


def test_batch_validate_job_saves_summary_and_continues_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(job_manager, "JOB_RESULTS_DIR", tmp_path / "jobs" / "results")

    entries = [
        {"gap_id": "research_gap_1", "gap_type": "research_gap", "gap": "Gap one"},
        {"gap_id": "methodological_gap_1", "gap_type": "methodological_gap", "gap": "Gap two"},
    ]
    monkeypatch.setattr(
        job_manager.gap_validator,
        "find_existing_gap_analysis_for_project",
        lambda project: {"analysis": {"research_gaps": [{"gap": "Gap one"}]}, "path": "gap.json"},
    )
    monkeypatch.setattr(job_manager.gap_validator, "_gap_entries", lambda analysis: entries)

    def fake_batch_validate_gaps(**kwargs):
        progress = kwargs["progress_callback"]
        progress(
            "completed",
            entries[0],
            {
                "gap_id": "research_gap_1",
                "gap_type": "research_gap",
                "original_gap": "Gap one",
                "status": "confirmed_candidate_gap",
                "confidence": "medium",
                "use_for_experiments": True,
                "refined_gap": None,
                "artifact_path": str(tmp_path / "one.json"),
            },
            None,
        )
        progress("failed", entries[1], None, "search failed")
        batch_path = tmp_path / "batch.json"
        result = {
            "project": kwargs["project"],
            "gap_count": 2,
            "validated_count": 1,
            "failed_count": 1,
            "status_counts": {"confirmed_candidate_gap": 1},
            "validated_gaps": [],
            "failed_gaps": [{"gap_id": "methodological_gap_1", "error": "search failed"}],
            "batch_artifact_path": str(batch_path),
        }
        batch_path.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(job_manager.gap_validator, "batch_validate_gaps", fake_batch_validate_gaps)

    started = job_manager.start_batch_validate_gaps_job("proj", max_workers=2)
    status = _wait_for(started["job_id"])
    assert status["status"] == "completed"
    assert status["completed"] == 1
    assert status["failed"] == 1

    result = job_manager.get_job_result(started["job_id"])
    assert result["result"]["gap_count"] == 2
    assert result["result"]["validated_count"] == 1
    assert result["result"]["failed_count"] == 1
    assert result["result"]["batch_artifact_path"].endswith("batch.json")


def test_job_tool_wrappers_are_thin(monkeypatch):
    monkeypatch.setattr(
        jobs_tool_module,
        "start_batch_build_profiles_job",
        lambda **kwargs: {"status": "started", "job_id": "j1"},
    )
    monkeypatch.setattr(jobs_tool_module, "log_invocation", lambda *_args, **_kwargs: None)
    assert jobs_tool_module.start_batch_build_profiles_job_tool(
        [{"paper_id": "p1", "source": "arxiv"}]
    )["job_id"] == "j1"

    monkeypatch.setattr(
        jobs_tool_module,
        "get_job_status",
        lambda job_id: {"job_id": job_id, "status": "completed"},
    )
    assert jobs_tool_module.get_job_status_tool("j1")["status"] == "completed"
