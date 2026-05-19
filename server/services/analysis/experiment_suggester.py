"""LLM-powered experiment suggestion based on gap analysis."""
import json
import os
from copy import deepcopy
from datetime import datetime

import httpx

from config import DATA_DIR
from config import IONOS_MODEL, SUGGEST_EXPERIMENTS_TIMEOUT
from services.extraction.llm_client import LLMClient
from services.paper_repository import load_paper_metadata
from services.paper_repository import load_profile
from services.paper_repository import load_profile_or_insights
from services.analysis.gap_detector import detect_gaps
from services.analysis.gap_validator import (
    find_latest_batch_validation_summary,
    validated_gap_analysis_for_experiments,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_ANALYSIS_DIR = DATA_DIR / "analysis"


def _log_raw_llm_output() -> bool:
    return os.environ.get("LOG_RAW_LLM_OUTPUT", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }

_EXPERIMENT_SYSTEM_PROMPT = """\
You are a research advisor. Given an analysis of research gaps across multiple papers and short paper metadata, suggest concrete experiments that could address the identified gaps.

Rules:
- Each experiment must directly address a specific identified gap.
- Each experiment must cite the paper_ids or gap paper lists that motivate it.
- Only cite paper_ids that appear in the provided active paper set. Never cite outside papers.
- If support is unclear, use an empty builds_on list rather than inventing attribution.
- Experiments must be feasible for a research team with standard compute resources (not "train a 100B model from scratch").
- Propose specific methods, baselines, and evaluation criteria.
- Reference actual paper_ids when building on existing work.
- Rate feasibility honestly: "high" means doable in weeks, "medium" in months, "low" means requires significant resources or novel techniques.
- Suggest 3-5 experiments, prioritized by impact and feasibility.
- Do NOT suggest vague directions like "explore more". Each experiment must have a testable hypothesis.

Return ONLY a valid JSON object with this structure:
{
  "experiments": [
    {
      "title": "short experiment name",
      "addresses_gap": "which gap this tackles",
      "hypothesis": "what you would test",
      "method": "proposed approach in 2-3 sentences",
      "baselines": ["what to compare against"],
      "datasets": ["suggested evaluation data"],
      "expected_outcome": "what success looks like",
      "feasibility": "high or medium or low",
      "builds_on": ["paper_ids this extends"]
    }
  ]
}
Do not include any text outside the JSON object.\
"""

_EXPERIMENT_USER_TEMPLATE = """\
Here is a gap analysis across {n} research papers, followed by short metadata for the active paper set.

--- GAP ANALYSIS ---
{gap_analysis_text}

--- ACTIVE PAPER SET ---
{paper_metadata_text}

Suggest concrete experiments to address the identified gaps. Return the JSON object now.\
"""


def _paper_id_set(papers: list[dict]) -> set[str]:
    return {p.get("paper_id") for p in papers if p.get("paper_id")}


def _safe_artifact_id(paper_id: str) -> str:
    return str(paper_id or "?").replace("/", "-")


def _matching_papers(left: list[dict], right: list[dict]) -> bool:
    left_keys = sorted((p.get("source"), p.get("paper_id")) for p in left)
    right_keys = sorted((p.get("source"), p.get("paper_id")) for p in right)
    return left_keys == right_keys


def find_existing_gap_analysis(papers: list[dict]) -> dict | None:
    """Return the newest gap analysis artifact matching exactly this paper set."""
    for path in sorted(_ANALYSIS_DIR.glob("gap_analysis_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if _matching_papers(data.get("papers", []), papers):
                return {
                    "analysis": data.get("analysis", {}),
                    "path": str(path),
                    "papers": data.get("papers", []),
                }
        except Exception:
            logger.warning("Could not inspect gap analysis artifact: %s", path)
    return None


def save_gap_analysis(papers: list[dict], analysis: dict) -> str:
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(_safe_artifact_id(ref.get("paper_id")) for ref in papers[:3])
    path = _ANALYSIS_DIR / f"gap_analysis_{timestamp}_{paper_ids}.json"
    path.write_text(
        json.dumps({"papers": papers, "analysis": analysis}, indent=2),
        encoding="utf-8",
    )
    logger.info("Gap analysis saved to %s", path)
    return str(path)


def save_experiment_suggestions(
    papers: list[dict],
    gap_analysis: dict,
    experiments: dict,
    gap_analysis_path: str | None = None,
    validation_metadata: dict | None = None,
) -> str:
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(_safe_artifact_id(ref.get("paper_id")) for ref in papers[:3])
    path = _ANALYSIS_DIR / f"experiments_{timestamp}_{paper_ids}.json"
    payload = {
        "papers": papers,
        "gaps": gap_analysis,
        "gap_analysis_path": gap_analysis_path,
        "validation": validation_metadata or {},
        "experiments": experiments,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Experiment suggestions saved to %s", path)
    return str(path)


def _filter_gap_analysis(gap_analysis: dict, allowed_ids: set[str]) -> dict:
    result = {
        "research_gaps": deepcopy(gap_analysis.get("research_gaps") or []),
        "methodological_gaps": deepcopy(gap_analysis.get("methodological_gaps") or []),
        "contradictions": deepcopy(gap_analysis.get("contradictions") or []),
        "field_summary": gap_analysis.get("field_summary") or "",
    }

    for gap in result["research_gaps"]:
        gap["relevant_papers"] = [
            pid for pid in gap.get("relevant_papers", []) if pid in allowed_ids
        ]
    for gap in result["methodological_gaps"]:
        gap["relevant_papers"] = [
            pid for pid in gap.get("relevant_papers", []) if pid in allowed_ids
        ]
    for contradiction in result["contradictions"]:
        if contradiction.get("paper_a") not in allowed_ids:
            contradiction["paper_a"] = ""
        if contradiction.get("paper_b") not in allowed_ids:
            contradiction["paper_b"] = ""
    return result


def _metadata_for_papers(
    papers: list[dict],
    paper_profiles: list[dict] | None = None,
) -> list[dict]:
    profiles_by_key = {
        (p.get("source"), p.get("paper_id")): p
        for p in paper_profiles or []
    }
    metadata = []
    for ref in papers:
        source = ref.get("source")
        paper_id = ref.get("paper_id")
        stored = load_paper_metadata(source, paper_id) or {}
        profile = profiles_by_key.get((source, paper_id), {})
        metadata.append(
            {
                "paper_id": paper_id,
                "source": source,
                "title": stored.get("title") or profile.get("title") or "",
                "year": stored.get("year") or profile.get("year") or "",
            }
        )
    return metadata


def _format_profile(profile: dict) -> str:
    return (
        f"--- Paper: {profile.get('paper_id', '?')} ({profile.get('source', '?')}) ---\n"
        f"Type: {profile.get('paper_type', '')}\n"
        f"Problem: {profile.get('research_problem', '')}\n"
        f"Contribution: {profile.get('main_contribution', '')}\n"
        f"Methods: {profile.get('methods_or_approach', '')}\n"
        f"Findings: {profile.get('key_findings', '')}\n"
        f"Limitations: {json.dumps(profile.get('limitations', []))}"
    )


def _filter_experiment_ids(result: dict, allowed_ids: set[str]) -> dict:
    for experiment in result.get("experiments", []):
        experiment["builds_on"] = [
            pid for pid in experiment.get("builds_on", []) if pid in allowed_ids
        ]
    return result


def suggest_experiments(
    gap_analysis: dict,
    paper_profiles: list[dict] | None = None,
    *,
    papers: list[dict] | None = None,
    compact: bool = True,
    paper_metadata: list[dict] | None = None,
) -> tuple[dict, str]:
    active_papers = papers or paper_profiles or []
    allowed_ids = _paper_id_set(active_papers)
    compact_gap_analysis = _filter_gap_analysis(gap_analysis, allowed_ids)
    gap_text = json.dumps(
        compact_gap_analysis,
        indent=2,
    )

    if compact:
        metadata = paper_metadata or _metadata_for_papers(active_papers, paper_profiles)
        paper_context_text = json.dumps(metadata, indent=2)
    else:
        paper_context_text = "\n\n".join(_format_profile(p) for p in paper_profiles or [])

    user_message = _EXPERIMENT_USER_TEMPLATE.format(
        n=len(active_papers),
        gap_analysis_text=gap_text,
        paper_metadata_text=paper_context_text,
    )

    logger.info(
        "Suggesting experiments: model=%s papers=%d compact=%s input_chars=%d gap_chars=%d",
        IONOS_MODEL, len(active_papers), compact, len(user_message), len(gap_text),
    )

    parsed, raw = LLMClient().call(
        system=_EXPERIMENT_SYSTEM_PROMPT,
        user=user_message,
        json_mode=True,
        timeout=SUGGEST_EXPERIMENTS_TIMEOUT,
        tool_name="suggest_experiments",
        input_chars=len(user_message),
    )

    logger.info("Experiment suggestion LLM response: %d chars", len(raw))
    if _log_raw_llm_output():
        logger.debug("Raw experiment completion: %s", raw)
    logger.info(
        "Experiment suggestions complete: count=%d",
        len(parsed.get("experiments", [])),
    )
    return _filter_experiment_ids(parsed, allowed_ids), raw


def suggest_experiments_for_papers(
    papers: list[dict],
    gap_analysis: dict | None = None,
    *,
    compact: bool = True,
    project: str | None = None,
) -> dict:
    """Generate experiment suggestions for an exact active paper set."""
    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for experiment suggestions.")

    gap_analysis_path = None
    gap_source = "provided"
    profiles = None
    validation_metadata = {
        "validation_used": False,
        "batch_validation_path": None,
        "included_gap_count": None,
        "excluded_gap_count": 0,
        "excluded_gaps": [],
        "refined_gap_count": 0,
        "validation_recommendation": None,
    }

    if gap_analysis is None:
        cached = find_existing_gap_analysis(papers)
        if cached is not None:
            gap_analysis = cached["analysis"]
            gap_analysis_path = cached["path"]
            gap_source = "cache"
            logger.info("Gap analysis cache hit: %s", gap_analysis_path)
        else:
            gap_source = "generated"
            logger.info("Gap analysis cache miss; running detection for %d papers", len(papers))
            profiles = [
                load_profile_or_insights(ref.get("source"), ref.get("paper_id"))
                for ref in papers
            ]
            gap_analysis, _ = detect_gaps(profiles)
            gap_analysis_path = save_gap_analysis(papers, gap_analysis)
    else:
        logger.info("Using provided gap analysis (skipping detection)")

    if compact and project:
        batch_validation = find_latest_batch_validation_summary(project, papers=papers)
        if batch_validation is not None:
            validated_gap_analysis, validation_metadata = validated_gap_analysis_for_experiments(
                batch_validation
            )
            gap_analysis = validated_gap_analysis
            if validation_metadata["included_gap_count"] > 0:
                logger.info(
                    "Using batch validation summary for experiments: %s included_gaps=%d excluded_gaps=%d",
                    validation_metadata.get("batch_validation_path"),
                    validation_metadata["included_gap_count"],
                    validation_metadata["excluded_gap_count"],
                )
            else:
                validation_metadata["validation_recommendation"] = (
                    "Batch validation found no gaps marked use_for_experiments=true; experiment prompt will use an empty validated gap set."
                )
                logger.warning(validation_metadata["validation_recommendation"])
        else:
            validation_metadata["validation_recommendation"] = (
                "No batch validation summary found; proceeding with unvalidated gaps. "
                "Run batch_validate_gaps_tool(project=..., max_workers=2) to use validated/refined gaps."
            )
            logger.info(validation_metadata["validation_recommendation"])

    if not compact and profiles is None:
        profiles = [
            load_profile(ref.get("source"), ref.get("paper_id")) or {}
            for ref in papers
        ]

    metadata = _metadata_for_papers(papers, profiles)
    try:
        result, raw = suggest_experiments(
            gap_analysis,
            profiles,
            papers=papers,
            compact=compact,
            paper_metadata=metadata,
        )
    except httpx.TimeoutException as e:
        logger.error("Experiment suggestion timed out: %s", e)
        return {
            "error": "Experiment suggestion LLM call timed out.",
            "gap_analysis_path": gap_analysis_path,
            "gap_source": gap_source,
            "project": project,
            "active_paper_count": len(papers),
            "paper_count": len(papers),
            "subset_used": False,
            "compact": compact,
            "suggestion": "Retry with compact=True or fewer gaps. The full active paper set was not replaced with a smaller subset.",
            "gaps": gap_analysis,
            "experiments": [],
            "experiment_count": 0,
            "gap_count": len((gap_analysis or {}).get("research_gaps", [])),
            "status": "failed",
            **validation_metadata,
        }

    save_path = save_experiment_suggestions(
        papers,
        gap_analysis,
        result,
        gap_analysis_path=gap_analysis_path,
        validation_metadata=validation_metadata,
    )
    gap_count = (
        len((gap_analysis or {}).get("research_gaps", []))
        + len((gap_analysis or {}).get("methodological_gaps", []))
    )
    return {
        "gaps": gap_analysis,
        "experiments": result.get("experiments", []),
        "experiment_count": len(result.get("experiments", [])),
        "gap_count": gap_count,
        "gap_analysis_path": gap_analysis_path,
        "gap_source": gap_source,
        "project": project,
        "save_path": save_path,
        "active_paper_count": len(papers),
        "paper_count": len(papers),
        "subset_used": False,
        "compact": compact,
        "error": None,
        "status": "succeeded",
        **validation_metadata,
    }
