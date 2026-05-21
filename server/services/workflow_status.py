"""Workflow status helpers for modular research-agent orchestration."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import DATA_DIR, WORKFLOW_MAX_PAPERS
from services.paper_repository import load_paper_cache, load_profile
from services.project_manager import get_project
from utils.logger import get_logger

logger = get_logger(__name__)

_ANALYSIS_DIR = DATA_DIR / "analysis"
_VALIDATION_BATCH_DIR = DATA_DIR / "analysis" / "gap_validations" / "batches"
_BIBLIOGRAPHY_DIR = DATA_DIR / "artifacts" / "bibliographies"
_REPORT_DIR = DATA_DIR / "artifacts" / "reports"


def _safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (text or "").strip()).strip("-")
    return (slug[:max_len].strip("-") or "project")


def _same_project_name(left: str | None, right: str | None) -> bool:
    return _safe_slug(left or "").lower() == _safe_slug(right or "").lower()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read workflow artifact: %s", path)
        return None


def _matching_papers(left: list[dict], right: list[dict]) -> bool:
    left_keys = sorted((p.get("source"), p.get("paper_id")) for p in left or [])
    right_keys = sorted((p.get("source"), p.get("paper_id")) for p in right or [])
    return left_keys == right_keys


def _latest_json(paths: list[Path], predicate) -> tuple[dict | None, str]:
    for path in sorted(paths, reverse=True):
        data = _read_json(path)
        if data is not None and predicate(data):
            return data, str(path)
    return None, ""


def _latest_gap_analysis(papers: list[dict]) -> tuple[dict | None, str]:
    return _latest_json(
        list(_ANALYSIS_DIR.glob("gap_analysis_*.json")) + list(_ANALYSIS_DIR.glob("lc_gap_analysis_*.json")),
        lambda data: _matching_papers(data.get("papers", []), papers),
    )


def _latest_validation(project: str, papers: list[dict]) -> tuple[dict | None, str]:
    return _latest_json(
        list(_VALIDATION_BATCH_DIR.glob("batch_gap_validation_*.json")),
        lambda data: _same_project_name(data.get("project"), project)
        and _matching_papers(data.get("project_papers", []), papers),
    )


def _latest_experiments(papers: list[dict]) -> tuple[dict | None, str]:
    return _latest_json(
        list(_ANALYSIS_DIR.glob("experiments_*.json")),
        lambda data: _matching_papers(data.get("papers", []), papers),
    )


def _latest_file(directory: Path, pattern: str) -> str:
    matches = sorted(directory.glob(pattern), reverse=True)
    return str(matches[0]) if matches else ""


def _artifact_counts(
    gap_analysis: dict | None,
    validation: dict | None,
    experiments: dict | None,
) -> dict[str, Any]:
    analysis = (gap_analysis or {}).get("analysis") or gap_analysis or {}
    research_gaps = analysis.get("research_gaps") or []
    methodological_gaps = analysis.get("methodological_gaps") or []
    validated = (validation or {}).get("validated_gaps") or []
    exp_payload = (experiments or {}).get("experiments") if experiments else []
    if isinstance(exp_payload, dict):
        exp_payload = exp_payload.get("experiments") or []
    return {
        "gap_count": len(research_gaps) + len(methodological_gaps),
        "validated_gap_count": len(validated),
        "included_validated_gap_count": sum(1 for gap in validated if gap.get("use_for_experiments")),
        "excluded_validated_gap_count": sum(1 for gap in validated if gap.get("status") == "already_addressed"),
        "experiment_count": len(exp_payload or []),
    }


def _next_step(
    paper_count: int,
    ingested_count: int,
    profiled_count: int,
    has_gap_analysis: bool,
    has_validation: bool,
    has_experiments: bool,
    has_bibliography: bool,
    has_report: bool,
) -> dict:
    if paper_count == 0:
        return {
            "tool": "batch_ingest_papers_tool",
            "reason": "No project papers are present. Search, select papers, ingest them, then profile before adding.",
        }
    if ingested_count < paper_count:
        return {
            "tool": "batch_ingest_papers_tool",
            "reason": "Some project papers are not ingested. Ingest missing papers before profiling.",
        }
    if profiled_count < paper_count:
        return {
            "tool": "start_batch_build_profiles_job",
            "reason": "Some project papers do not have profiles. Build profiles, then add only profiled papers to the project.",
        }
    if not has_gap_analysis:
        return {"tool": "detect_gaps_tool", "reason": "Profiles exist; detect candidate research gaps next."}
    if not has_validation:
        return {"tool": "start_batch_validate_gaps_job", "reason": "Gap analysis exists; validate gaps before experiments."}
    if not has_experiments:
        return {"tool": "suggest_experiments_tool", "reason": "Validated gaps exist; generate grounded experiment suggestions."}
    if not has_bibliography:
        return {"tool": "generate_bibliography_tool", "reason": "Generate BibTeX before the final report."}
    if not has_report:
        return {"tool": "generate_project_report_tool", "reason": "All core artifacts exist; generate the final Markdown report."}
    return {"tool": "complete", "reason": "The workflow has all expected artifacts."}


def get_workflow_status(project: str) -> dict:
    """Return compact project workflow state and the next recommended tool."""
    manifest = get_project(project)
    project_name = manifest["name"]
    papers = manifest.get("papers") or []
    ingested = []
    profiled = []
    missing_ingest = []
    missing_profiles = []

    for ref in papers:
        paper_id = ref.get("paper_id") or ""
        source = ref.get("source") or ""
        item = {"paper_id": paper_id, "source": source}
        if paper_id and source and load_paper_cache(source, paper_id) is not None:
            ingested.append(item)
        else:
            missing_ingest.append(item)
        if paper_id and source and load_profile(source, paper_id) is not None:
            profiled.append(item)
        else:
            missing_profiles.append(item)

    gap_analysis, gap_path = _latest_gap_analysis(papers)
    validation, validation_path = _latest_validation(project_name, papers)
    experiments, experiments_path = _latest_experiments(papers)
    project_slug = _safe_slug(project_name)
    bibliography_path = (
        _latest_file(_BIBLIOGRAPHY_DIR, f"{project_slug}_*.bib")
        or _latest_file(_BIBLIOGRAPHY_DIR, f"{project_slug}_*.md")
    )
    report_path = _latest_file(_REPORT_DIR, f"{project_slug}_*_report.md")
    counts = _artifact_counts(gap_analysis, validation, experiments)
    warnings = []
    if len(papers) > WORKFLOW_MAX_PAPERS:
        warnings.append(f"Project has {len(papers)} papers, above the normal cap of {WORKFLOW_MAX_PAPERS}.")
    next_step = _next_step(
        paper_count=len(papers),
        ingested_count=len(ingested),
        profiled_count=len(profiled),
        has_gap_analysis=bool(gap_path),
        has_validation=bool(validation_path),
        has_experiments=bool(experiments_path),
        has_bibliography=bool(bibliography_path),
        has_report=bool(report_path),
    )

    return {
        "project": project_name,
        "paper_count": len(papers),
        "ingested_count": len(ingested),
        "profiled_count": len(profiled),
        "missing_ingest": missing_ingest,
        "missing_profiles": missing_profiles,
        "artifacts": {
            "gap_analysis_path": gap_path,
            "validation_batch_path": validation_path,
            "experiments_path": experiments_path,
            "bibliography_path": bibliography_path,
            "report_path": report_path,
        },
        "counts": counts,
        "warnings": warnings,
        "next_step": next_step,
    }
