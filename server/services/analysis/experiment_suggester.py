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
- Each experiment should build on at least one active project paper when possible.
- If no active project paper supports an experiment, mark the idea narrowly and avoid presenting it as strongly grounded.
- Experiments must be feasible for a research team with standard compute resources (not "train a 100B model from scratch").
- Propose specific methods, baselines, and evaluation criteria.
- Reference actual paper_ids when building on existing work.
- Do not use outside validation papers as builds_on references.
- Rate feasibility honestly: "high" means doable in weeks, "medium" in months, "low" means requires significant resources or novel techniques.
- Suggest no more experiments than requested, prioritized by impact and feasibility.
- Do not create multiple experiments that only differ by wording, dataset, or scope label.
- Prefer fewer, more distinct experiments when the validated gap set is small.
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

Requested experiment count: {requested_experiment_count}

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
    paths = list(_ANALYSIS_DIR.glob("gap_analysis_*.json")) + list(_ANALYSIS_DIR.glob("lc_gap_analysis_*.json"))
    for path in sorted(paths, reverse=True):
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


def _profile_text(profile: dict) -> str:
    return " ".join(
        str(profile.get(key) or "")
        for key in (
            "paper_id",
            "paper_type",
            "research_problem",
            "main_contribution",
            "methods_or_approach",
            "key_findings",
            "future_work",
            "limitations",
            "plain_english_summary",
        )
    )


def _load_grounding_profiles(papers: list[dict], paper_profiles: list[dict] | None = None) -> list[dict]:
    provided = {
        (profile.get("source"), profile.get("paper_id")): profile
        for profile in paper_profiles or []
    }
    profiles = []
    for ref in papers:
        key = (ref.get("source"), ref.get("paper_id"))
        profile = provided.get(key)
        if profile is None:
            stored = load_paper_metadata(ref.get("source"), ref.get("paper_id")) or {}
            profile = {
                "paper_id": ref.get("paper_id"),
                "source": ref.get("source"),
                "research_problem": stored.get("title", ""),
                "main_contribution": stored.get("title", ""),
            }
        if profile:
            profiles.append(profile)
    return profiles


def _best_grounding_ids(experiment: dict, papers: list[dict], profiles: list[dict]) -> list[str]:
    query_tokens = _content_tokens(
        " ".join(
            str(experiment.get(key) or "")
            for key in ("title", "addresses_gap", "hypothesis", "method")
        )
    )
    if not query_tokens:
        return []
    scored = []
    for profile in profiles:
        pid = profile.get("paper_id")
        if not pid:
            continue
        tokens = _content_tokens(_profile_text(profile))
        score = _similarity(query_tokens, tokens)
        if score > 0:
            scored.append((score, pid))
    scored.sort(reverse=True)
    if scored:
        return [pid for score, pid in scored[:3] if score >= 0.06] or [scored[0][1]]
    return [ref.get("paper_id") for ref in papers[:1] if ref.get("paper_id")]


def _ground_experiments(result: dict, papers: list[dict], paper_profiles: list[dict] | None = None) -> tuple[dict, dict]:
    profiles = _load_grounding_profiles(papers, paper_profiles)
    weak_count = 0
    filled_count = 0
    for experiment in result.get("experiments") or []:
        flags = list(experiment.get("quality_flags") or [])
        if not experiment.get("builds_on"):
            inferred = _best_grounding_ids(experiment, papers, profiles)
            if inferred:
                experiment["builds_on"] = inferred
                experiment["grounding_status"] = "inferred_project_papers"
                filled_count += 1
                flags.append("inferred_project_grounding")
            else:
                experiment["grounding_status"] = "weak"
                experiment["warning"] = "No direct project paper grounding found."
                flags.append("missing_project_grounding")
                weak_count += 1
        else:
            experiment["grounding_status"] = "project_papers"
        if not experiment.get("baselines"):
            flags.append("missing_baselines")
        if not experiment.get("datasets"):
            flags.append("missing_dataset")
        experiment["quality_flags"] = sorted(set(flags))
    return result, {
        "grounding_inferred_count": filled_count,
        "weakly_grounded_experiment_count": weak_count,
    }


def _gap_count(gap_analysis: dict | None) -> int:
    return (
        len((gap_analysis or {}).get("research_gaps", []))
        + len((gap_analysis or {}).get("methodological_gaps", []))
    )


def _requested_experiment_count(included_gap_count: int | None) -> int:
    if included_gap_count is None:
        return 5
    if included_gap_count <= 0:
        return 0
    if included_gap_count == 1:
        return 2
    if included_gap_count == 2:
        return 4
    return 5


def _normalize_text(value: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _content_tokens(value: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "it",
        "of", "on", "or", "the", "to", "under", "with", "without",
    }
    return {token for token in _normalize_text(value).split() if token not in stop and len(token) > 2}


def _method_keywords(experiment: dict) -> set[str]:
    text = " ".join(
        str(experiment.get(key) or "")
        for key in ("title", "addresses_gap", "hypothesis", "method")
    )
    keywords = {
        "explainability", "interpretability", "shap", "lime", "prompt", "injection",
        "tool", "tools", "workflow", "multi", "agent", "users", "human", "benchmark",
        "simulation", "deployment", "detector", "detection", "mitigation", "warning",
    }
    return _content_tokens(text) & keywords


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _experiment_quality(experiment: dict) -> tuple[int, int, int, int]:
    feasibility_rank = {"high": 3, "medium": 2, "low": 1}
    method_len = len(str(experiment.get("method") or ""))
    hypothesis_len = len(str(experiment.get("hypothesis") or ""))
    builds_on_len = len(experiment.get("builds_on") or [])
    feasibility = feasibility_rank.get(str(experiment.get("feasibility") or "").lower(), 0)
    return (method_len + hypothesis_len, builds_on_len, feasibility, len(str(experiment.get("title") or "")))


def _experiments_are_duplicates(left: dict, right: dict) -> bool:
    left_title = _normalize_text(left.get("title", ""))
    right_title = _normalize_text(right.get("title", ""))
    if left_title and left_title == right_title:
        return True

    left_gap = _normalize_text(left.get("addresses_gap", ""))
    right_gap = _normalize_text(right.get("addresses_gap", ""))
    same_gap = bool(left_gap and right_gap and left_gap == right_gap)
    method_similarity = _similarity(_method_keywords(left), _method_keywords(right))
    hypothesis_similarity = _similarity(
        _content_tokens(left.get("hypothesis", "")),
        _content_tokens(right.get("hypothesis", "")),
    )
    title_similarity = _similarity(_content_tokens(left.get("title", "")), _content_tokens(right.get("title", "")))
    return same_gap and (method_similarity >= 0.55 or hypothesis_similarity >= 0.45 or title_similarity >= 0.55)


def _dedupe_experiments(result: dict, requested_count: int) -> tuple[dict, dict]:
    experiments = result.get("experiments") or []
    deduped: list[dict] = []
    for experiment in experiments:
        duplicate_index = None
        for index, kept in enumerate(deduped):
            if _experiments_are_duplicates(experiment, kept):
                duplicate_index = index
                break
        if duplicate_index is None:
            deduped.append(experiment)
        elif _experiment_quality(experiment) > _experiment_quality(deduped[duplicate_index]):
            deduped[duplicate_index] = experiment

    final = deduped[:requested_count] if requested_count > 0 else []
    output = dict(result)
    output["experiments"] = final
    metadata = {
        "requested_experiment_count": requested_count,
        "raw_experiment_count": len(experiments),
        "deduplicated_experiment_count": len(deduped),
        "final_experiment_count": len(final),
    }
    return output, metadata


def suggest_experiments(
    gap_analysis: dict,
    paper_profiles: list[dict] | None = None,
    *,
    papers: list[dict] | None = None,
    compact: bool = True,
    paper_metadata: list[dict] | None = None,
    requested_experiment_count: int = 5,
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
        requested_experiment_count=requested_experiment_count,
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
    filtered = _filter_experiment_ids(parsed, allowed_ids)
    grounded, grounding_metadata = _ground_experiments(filtered, active_papers, paper_profiles)
    deduped, dedupe_metadata = _dedupe_experiments(grounded, requested_experiment_count)
    deduped["_meta"] = {**deduped.get("_meta", {}), **dedupe_metadata, **grounding_metadata}
    return deduped, raw


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
                    "No experiments were generated because all detected gaps were validated as already addressed by external literature. "
                    "Consider expanding the paper set, detecting more specific gaps, or manually refining candidate gaps."
                )
                logger.warning(
                    "No validated gaps available for experiment generation; skipping LLM call."
                )
        else:
            validation_metadata["validation_recommendation"] = (
                "No batch validation summary found; proceeding with unvalidated gaps. "
                "Run start_batch_validate_gaps_job(project=..., max_workers=2) to use validated/refined gaps."
            )
            logger.info(validation_metadata["validation_recommendation"])

    if (
        compact
        and project
        and validation_metadata.get("validation_used")
        and validation_metadata.get("included_gap_count") == 0
    ):
        gap_count = _gap_count(gap_analysis)
        empty_result = {
            "experiments": [],
            "_meta": {
                "requested_experiment_count": 0,
                "raw_experiment_count": 0,
                "deduplicated_experiment_count": 0,
                "final_experiment_count": 0,
            },
        }
        save_path = save_experiment_suggestions(
            papers,
            gap_analysis,
            empty_result,
            gap_analysis_path=gap_analysis_path,
            validation_metadata=validation_metadata,
        )
        return {
            "gaps": gap_analysis,
            "experiments": [],
            "experiment_count": 0,
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
            "status": "no_validated_gaps",
            "recommended_next_step": validation_metadata["validation_recommendation"],
            **empty_result["_meta"],
            **validation_metadata,
        }

    if not compact and profiles is None:
        profiles = [
            load_profile(ref.get("source"), ref.get("paper_id")) or {}
            for ref in papers
        ]

    metadata = _metadata_for_papers(papers, profiles)
    try:
        requested_count = _requested_experiment_count(
            validation_metadata.get("included_gap_count")
            if validation_metadata.get("validation_used")
            else _gap_count(gap_analysis)
        )
        result, raw = suggest_experiments(
            gap_analysis,
            profiles,
            papers=papers,
            compact=compact,
            paper_metadata=metadata,
            requested_experiment_count=requested_count,
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
    gap_count = _gap_count(gap_analysis)
    experiment_meta = result.get("_meta", {})
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
        **experiment_meta,
        **validation_metadata,
    }
