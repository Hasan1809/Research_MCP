"""Deterministic Markdown report generation from saved research artifacts."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATA_DIR
from services.citations import generate_bibliography
from services.paper_repository import load_paper_metadata
from services.project_manager import get_project
from utils.logger import get_logger

logger = get_logger(__name__)

_ANALYSIS_DIR = DATA_DIR / "analysis"
_VALIDATION_BATCH_DIR = DATA_DIR / "analysis" / "gap_validations" / "batches"
_BIBLIOGRAPHY_DIR = DATA_DIR / "artifacts" / "bibliographies"
_REPORT_DIR = DATA_DIR / "artifacts" / "reports"
_MAX_BIBTEX_INLINE_CHARS = 20000


def _safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (text or "").strip()).strip("-")
    return (slug[:max_len].strip("-") or "project")


def _read_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read JSON artifact: %s", path)
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


def _find_latest_gap_analysis(project_papers: list[dict]) -> tuple[dict | None, str]:
    return _latest_json(
        list(_ANALYSIS_DIR.glob("gap_analysis_*.json")) + list(_ANALYSIS_DIR.glob("lc_gap_analysis_*.json")),
        lambda data: _matching_papers(data.get("papers", []), project_papers),
    )


def _find_latest_validation(project: str, project_papers: list[dict]) -> tuple[dict | None, str]:
    return _latest_json(
        list(_VALIDATION_BATCH_DIR.glob("batch_gap_validation_*.json")),
        lambda data: data.get("project") == project
        and _matching_papers(data.get("project_papers", []), project_papers),
    )


def _find_latest_experiments(project_papers: list[dict]) -> tuple[dict | None, str]:
    return _latest_json(
        list(_ANALYSIS_DIR.glob("experiments_*.json")),
        lambda data: _matching_papers(data.get("papers", []), project_papers),
    )


def _find_latest_bibliography(project: str) -> str:
    project_slug = _safe_slug(project)
    matches = sorted(_BIBLIOGRAPHY_DIR.glob(f"{project_slug}_*.bib"), reverse=True)
    if matches:
        return str(matches[0])
    matches = sorted(_BIBLIOGRAPHY_DIR.glob(f"{project_slug}_*.md"), reverse=True)
    return str(matches[0]) if matches else ""


def _metadata_for_paper(ref: dict) -> dict:
    paper_id = ref.get("paper_id") or ""
    source = ref.get("source") or ""
    metadata = load_paper_metadata(source, paper_id) if paper_id and source else None
    return metadata or {
        "paper_id": paper_id,
        "source": source,
        "title": paper_id,
        "year": "",
    }


def _metadata_warning_count(papers: list[dict]) -> int:
    count = 0
    for ref in papers:
        metadata = _metadata_for_paper(ref)
        if metadata.get("metadata_warnings") or metadata.get("metadata_quality") in {"fallback", "pdf_extracted"}:
            count += 1
    return count


def _value(value: Any, default: str = "Not available") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    if isinstance(value, list):
        if not value:
            return default
        return ", ".join(str(item) for item in value)
    return str(value)


def _gap_analysis_payload(data: dict | None) -> dict:
    if not data:
        return {}
    return data.get("analysis") or data.get("gaps") or data


def _experiments_payload(data: dict | None) -> list[dict]:
    if not data:
        return []
    experiments = data.get("experiments")
    if isinstance(experiments, dict):
        return experiments.get("experiments") or []
    if isinstance(experiments, list):
        return experiments
    return []


def _artifact_path(data: dict | None, fallback: str) -> str:
    if not data:
        return fallback or "Not available"
    return (
        data.get("artifact_path")
        or data.get("batch_artifact_path")
        or fallback
        or "Not available"
    )


def _format_evidence_item(item: dict) -> str:
    title = _value(item.get("title"))
    classification = _value(item.get("classification"))
    score = item.get("score")
    score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else _value(score)
    parts = []
    if item.get("year"):
        parts.append(str(item["year"]))
    if item.get("source"):
        parts.append(str(item["source"]))
    parts.extend([classification, score_text])
    return f"{title} - {' - '.join(parts)}"


def _append_gap_list(lines: list[str], title: str, gaps: list[dict]) -> None:
    lines.append(f"### {title}")
    if not gaps:
        lines.append("- Not available")
        lines.append("")
        return
    for index, gap in enumerate(gaps, start=1):
        lines.append(f"{index}. {_value(gap.get('gap') or gap.get('original_gap'))}")
        evidence = gap.get("evidence") or gap.get("missing_approach")
        if evidence:
            lines.append(f"   - Summary: {_value(evidence)}")
    lines.append("")


def _append_validation_section(lines: list[str], validation: dict | None) -> tuple[int, int, bool]:
    lines.append("## Gap Validation Summary")
    if not validation:
        lines.append("Not available")
        lines.append("")
        return 0, 0, False

    validated = validation.get("validated_gaps") or []
    included = [
        gap for gap in validated
        if gap.get("use_for_experiments")
        or gap.get("status") in {"partially_addressed", "needs_refinement", "confirmed_candidate_gap"}
    ]
    excluded = [gap for gap in validated if gap.get("status") == "already_addressed"]

    lines.append("### Gaps Used for Experiments")
    if not included:
        lines.append("- Not available")
    for gap in included:
        lines.append(f"- Original gap: {_value(gap.get('original_gap'))}")
        lines.append(f"  - Validation status: {_value(gap.get('status'))}")
        lines.append(f"  - Confidence: {_value(gap.get('confidence'))}")
        if gap.get("refined_gap"):
            lines.append(f"  - Refined gap: {_value(gap.get('refined_gap'))}")
        lines.append(f"  - Reason: {_value(gap.get('decision_reason'))}")
    lines.append("")

    lines.append("### Gaps Excluded from Experiments")
    if not excluded:
        lines.append("- Not available")
    for gap in excluded:
        lines.append(f"- Original gap: {_value(gap.get('original_gap'))}")
        lines.append(f"  - Validation status: {_value(gap.get('status'))}")
        lines.append(f"  - Confidence: {_value(gap.get('confidence'))}")
        reason = gap.get("decision_reason")
        evidence = gap.get("evidence_summary") or gap.get("top_evidence") or gap.get("external_evidence_titles") or []
        if reason:
            lines.append(f"  - Reason: {_value(reason)}")
        if evidence:
            lines.append("  - Evidence:")
            for item in evidence:
                if isinstance(item, dict):
                    lines.append(f"    - {_format_evidence_item(item)}")
                else:
                    lines.append(f"    - {_value(item)}")
    lines.append("")
    weak_warning = any(
        gap.get("evidence_quality_warning")
        or any(
            isinstance(item, dict) and item.get("classification") != "directly_addresses_gap"
            for item in (gap.get("evidence_summary") or gap.get("top_evidence") or [])
        )
        for gap in excluded
    )
    return len(included), len(excluded), weak_warning


def _append_experiments(lines: list[str], experiments: list[dict]) -> None:
    lines.append("## Suggested Experiments")
    if not experiments:
        lines.append("Not available")
        lines.append("")
        return
    for index, experiment in enumerate(experiments, start=1):
        lines.append(f"### Experiment {index}: {_value(experiment.get('title'), f'Experiment {index}')}")
        for label, key in (
            ("Hypothesis", "hypothesis"),
            ("Method", "method"),
            ("Feasibility", "feasibility"),
            ("Builds on", "builds_on"),
            ("Gap addressed", "addresses_gap"),
        ):
            if key in experiment:
                lines.append(f"- {label}: {_value(experiment.get(key))}")
        if experiment.get("grounding_status"):
            lines.append(f"- Grounding: {_value(experiment.get('grounding_status'))}")
        if experiment.get("quality_flags"):
            lines.append(f"- Quality flags: {_value(experiment.get('quality_flags'))}")
        lines.append("")


def _weakly_grounded_experiment_count(experiments: list[dict]) -> int:
    return sum(
        1 for experiment in experiments
        if not experiment.get("builds_on")
        or experiment.get("grounding_status") == "weak"
        or "missing_project_grounding" in (experiment.get("quality_flags") or [])
    )


def _append_bibliography(lines: list[str], bibliography_path: str, include_bibliography: bool) -> None:
    lines.append("## Bibliography")
    lines.append(f"- Bibliography file: {_value(bibliography_path)}")
    if include_bibliography and bibliography_path and Path(bibliography_path).exists():
        content = Path(bibliography_path).read_text(encoding="utf-8")
        if len(content) <= _MAX_BIBTEX_INLINE_CHARS:
            lines.append("")
            lines.append("```bibtex")
            lines.append(content)
            lines.append("```")
    lines.append("")


def generate_project_report(
    project: str,
    format: str = "markdown",
    gap_analysis_path: str | None = None,
    validation_batch_path: str | None = None,
    experiments_path: str | None = None,
    bibliography_path: str | None = None,
    include_bibliography: bool = True,
) -> dict:
    """Generate a deterministic Markdown report from existing saved artifacts."""
    if format.lower() != "markdown":
        raise ValueError("generate_project_report currently supports only format='markdown'.")
    manifest = get_project(project)
    project_name = manifest["name"]
    papers = manifest.get("papers") or []

    gap_analysis, resolved_gap_path = (
        (_read_json(gap_analysis_path), gap_analysis_path or "")
        if gap_analysis_path else _find_latest_gap_analysis(papers)
    )
    validation, resolved_validation_path = (
        (_read_json(validation_batch_path), validation_batch_path or "")
        if validation_batch_path else _find_latest_validation(project_name, papers)
    )
    experiments_data, resolved_experiments_path = (
        (_read_json(experiments_path), experiments_path or "")
        if experiments_path else _find_latest_experiments(papers)
    )
    resolved_bibliography_path = bibliography_path or _find_latest_bibliography(project_name)
    generated_bibliography = False
    if include_bibliography and not resolved_bibliography_path and papers:
        try:
            bibliography = generate_bibliography(
                project_name=project_name,
                format="bibtex",
                save=True,
            )
            resolved_bibliography_path = bibliography.get("artifact_path") or ""
            generated_bibliography = bool(resolved_bibliography_path)
        except Exception as e:
            logger.warning("Could not auto-generate bibliography for report: %s", e)

    logger.info(
        "Report artifacts: gap=%s validation=%s experiments=%s bibliography=%s",
        bool(gap_analysis),
        bool(validation),
        bool(experiments_data),
        bool(resolved_bibliography_path),
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    analysis = _gap_analysis_payload(gap_analysis)
    research_gaps = analysis.get("research_gaps") or []
    methodological_gaps = analysis.get("methodological_gaps") or []
    contradictions = analysis.get("contradictions") or []
    connections = analysis.get("connections") or []
    experiments = _experiments_payload(experiments_data)
    metadata_warning_count = _metadata_warning_count(papers)
    weakly_grounded_count = _weakly_grounded_experiment_count(experiments)
    included_summary_count = (
        sum(1 for gap in validation.get("validated_gaps", []) if gap.get("use_for_experiments"))
        if validation else 0
    )
    excluded_summary_count = (
        sum(1 for gap in validation.get("validated_gaps", []) if gap.get("status") == "already_addressed")
        if validation else 0
    )

    lines = [
        "# Research Agent Report",
        "",
        "## Project Summary",
        f"- Project name: {project_name}",
        f"- Generated: {generated_at}",
        f"- Number of papers: {len(papers)}",
        f"- Gap analysis file: {_artifact_path(gap_analysis, resolved_gap_path)}",
        f"- Validation file: {_artifact_path(validation, resolved_validation_path)}",
        f"- Experiments file: {_artifact_path(experiments_data, resolved_experiments_path)}",
        f"- Bibliography file: {_value(resolved_bibliography_path)}",
        f"- Bibliography auto-generated: {generated_bibliography}",
        f"- Included validated gaps: {included_summary_count}",
        f"- Excluded validated gaps: {excluded_summary_count}",
        f"- Experiment count: {len(experiments)}",
        "",
        "## Papers Included",
    ]

    if not papers:
        lines.append("- Not available")
    for index, ref in enumerate(papers, start=1):
        metadata = _metadata_for_paper(ref)
        title = metadata.get("title") or ref.get("paper_id")
        year = metadata.get("year") or "n.d."
        source = ref.get("source") or metadata.get("source") or "unknown"
        paper_id = ref.get("paper_id") or metadata.get("paper_id") or "unknown"
        lines.append(f"{index}. {title} - {year} - {source} - {paper_id}")
    lines.extend([
        "",
        "## Quality Checks",
        f"- Project reused existing papers: {_value(manifest.get('reused_existing', False))}",
        f"- Bibliography metadata warnings: {metadata_warning_count}",
        f"- Experiments without project grounding: {weakly_grounded_count}",
        f"- Validation evidence warnings: {'1' if validation and any(gap.get('evidence_quality_warning') for gap in validation.get('validated_gaps', [])) else '0'}",
    ])
    lines.extend([
        "",
        "## Gap Detection Summary",
        f"- Research gaps: {len(research_gaps)}",
        f"- Methodological gaps: {len(methodological_gaps)}",
        f"- Contradictions: {len(contradictions)}",
        f"- Connections: {len(connections)}",
        "",
    ])
    _append_gap_list(lines, "Research Gaps", research_gaps)
    _append_gap_list(lines, "Methodological Gaps", methodological_gaps)
    included_count, excluded_count, weak_warning = _append_validation_section(lines, validation)
    _append_experiments(lines, experiments)
    _append_bibliography(lines, resolved_bibliography_path, include_bibliography)
    lines.extend([
        "## Notes",
        "The gaps in this report are candidate gaps generated from the selected project papers and validated through external literature search. They should not be treated as absolute field-level claims without manual review.",
    ])
    if weak_warning:
        lines.append("Some validation evidence may be adjacent rather than directly addressing the gap. Manual review is recommended before treating candidate gaps as closed.")
    lines.append("")

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORT_DIR / f"{_safe_slug(project_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Project report saved: %s", report_path)

    return {
        "project": project_name,
        "report_path": str(report_path),
        "paper_count": len(papers),
        "gap_count": len(research_gaps) + len(methodological_gaps),
        "included_validated_gap_count": included_count,
        "excluded_validated_gap_count": excluded_count,
        "experiment_count": len(experiments),
        "bibliography_path": resolved_bibliography_path or "",
        "error": None,
    }
