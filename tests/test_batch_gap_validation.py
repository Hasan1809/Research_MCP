import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.analysis import gap_validator
from tools import batch_validate_gaps as batch_validate_gaps_tool_module


def _papers():
    return [
        {"paper_id": "p1", "source": "test"},
        {"paper_id": "p2", "source": "test"},
    ]


def _gap_analysis():
    return {
        "research_gaps": [
            {"gap": "Gap one about real-world LLM agent security evaluation", "relevant_papers": ["p1"]},
            {"gap": "Gap two about explainability for secure agent tool use", "relevant_papers": ["p2"]},
        ],
        "methodological_gaps": [
            {
                "gap": "Method gap about standardized cross benchmark metrics",
                "current_approaches": [],
                "missing_approach": "cross benchmark metrics",
                "relevant_papers": ["p1", "p2"],
            }
        ],
        "contradictions": [],
        "connections": [],
        "field_summary": "summary",
    }


def _write_gap_artifact(tmp_path, papers, analysis):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    path = analysis_dir / "gap_analysis_20260518_test.json"
    path.write_text(json.dumps({"papers": papers, "analysis": analysis}), encoding="utf-8")
    return path


def _validation_result(gap, status, refined_gap=""):
    path = gap_validator._VALIDATION_DIR / f"{status}_{gap_validator._safe_slug(gap)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gap": gap,
        "project": "proj",
        "mode": "metadata_only",
        "search_queries": [],
        "results_found": 1,
        "relevant_results": 1,
        "status": status,
        "confidence": "medium",
        "decision_reason": f"{status} reason",
        "refined_gap": refined_gap,
        "validation_papers": [
            {
                "paper_id": "external",
                "source": "semantic_scholar",
                "title": f"Evidence for {gap}",
                "classification": "partially_addresses_gap",
                "already_in_project": False,
            }
        ],
        "recommended_next_step": "",
        "artifact_path": str(path),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_batch_validate_loads_cached_gap_analysis_and_validates_all_gaps(tmp_path, monkeypatch):
    papers = _papers()
    gap_artifact = _write_gap_artifact(tmp_path, papers, _gap_analysis())
    validation_dir = tmp_path / "validations"
    calls = []

    monkeypatch.setattr(gap_validator, "_ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", validation_dir)
    monkeypatch.setattr(gap_validator, "_BATCH_VALIDATION_DIR", validation_dir / "batches")
    monkeypatch.setattr(gap_validator, "get_project_papers", lambda project: papers)

    statuses = {
        "Gap one about real-world LLM agent security evaluation": "confirmed_candidate_gap",
        "Gap two about explainability for secure agent tool use": "already_addressed",
        "Method gap about standardized cross benchmark metrics": "partially_addressed",
    }

    def fake_validate_gap(gap, project=None, max_results=10, mode="metadata_only"):
        calls.append(gap)
        refined = "Refined method gap" if statuses[gap] == "partially_addressed" else ""
        return _validation_result(gap, statuses[gap], refined_gap=refined)

    monkeypatch.setattr(gap_validator, "validate_gap", fake_validate_gap)

    result = gap_validator.batch_validate_gaps("proj", max_workers=2)

    assert result["gap_analysis_path"] == str(gap_artifact)
    assert result["gap_count"] == 3
    assert result["validated_count"] == 3
    assert result["failed_count"] == 0
    assert len(calls) == 3
    assert os.path.exists(result["batch_artifact_path"])
    assert all(os.path.exists(item["artifact_path"]) for item in result["validated_gaps"])


def test_batch_validate_continues_when_one_gap_fails(tmp_path, monkeypatch):
    papers = _papers()
    _write_gap_artifact(tmp_path, papers, _gap_analysis())
    validation_dir = tmp_path / "validations"

    monkeypatch.setattr(gap_validator, "_ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", validation_dir)
    monkeypatch.setattr(gap_validator, "_BATCH_VALIDATION_DIR", validation_dir / "batches")
    monkeypatch.setattr(gap_validator, "get_project_papers", lambda project: papers)

    def fake_validate_gap(gap, project=None, max_results=10, mode="metadata_only"):
        if "explainability" in gap:
            raise RuntimeError("search failed")
        return _validation_result(gap, "confirmed_candidate_gap")

    monkeypatch.setattr(gap_validator, "validate_gap", fake_validate_gap)

    result = gap_validator.batch_validate_gaps("proj", max_workers=2)

    assert result["validated_count"] == 2
    assert result["failed_count"] == 1
    assert result["failed_gaps"][0]["error"] == "search failed"


def test_use_for_experiments_rules():
    assert gap_validator.gap_use_for_experiments("confirmed_candidate_gap") is True
    assert gap_validator.gap_use_for_experiments("partially_addressed") is True
    assert gap_validator.gap_use_for_experiments("needs_refinement", "refined") is True
    assert gap_validator.gap_use_for_experiments("needs_refinement", "") is False
    assert gap_validator.gap_use_for_experiments("already_addressed") is False
    assert gap_validator.gap_use_for_experiments("too_broad", "") is False
    assert gap_validator.gap_use_for_experiments("too_broad", "refined") is True
    assert gap_validator.gap_use_for_experiments("insufficient_evidence") is False


def test_batch_validate_tool_wrapper_returns_compact_summary(tmp_path, monkeypatch):
    result = {
        "project": "proj",
        "gap_count": 1,
        "validated_count": 1,
        "failed_count": 0,
        "status_counts": {"confirmed_candidate_gap": 1},
        "batch_artifact_path": str(tmp_path / "batch.json"),
    }
    monkeypatch.setattr(
        batch_validate_gaps_tool_module,
        "batch_validate_gaps",
        lambda **kwargs: result,
    )
    monkeypatch.setattr(
        batch_validate_gaps_tool_module,
        "log_invocation",
        lambda *_args, **_kwargs: None,
    )

    assert batch_validate_gaps_tool_module.batch_validate_gaps_tool("proj") == result
