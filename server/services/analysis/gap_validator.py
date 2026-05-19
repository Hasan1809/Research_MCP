"""Metadata-only validation for candidate research gaps."""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import DATA_DIR
from services.project_manager import get_project_papers
from services.retrieval.aggregator import fetch_papers
from utils.logger import get_logger

logger = get_logger(__name__)

_VALIDATION_DIR = DATA_DIR / "analysis" / "gap_validations"
_BATCH_VALIDATION_DIR = _VALIDATION_DIR / "batches"
_ANALYSIS_DIR = DATA_DIR / "analysis"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "for", "from", "in", "into", "is",
    "lack", "limited", "missing", "no", "not", "of", "on", "or", "the", "to",
    "under", "with", "without",
}
_DOMAIN_TERMS = {
    "agent", "agents", "llm", "llms", "language", "model", "models", "security",
    "safety", "tool", "tools", "prompt", "injection", "defense", "defences",
    "defenses", "evaluation", "benchmark", "real", "world", "deployment",
    "production",
}
_DIRECT_TERMS = {
    "evaluate", "evaluates", "evaluation", "study",
    "studies", "experiment", "experiments", "empirical", "deployment",
    "deployed", "production", "real-world", "field", "longitudinal",
}
_PARTIAL_TERMS = {
    "framework", "survey", "taxonomy", "analysis", "dataset", "benchmark",
    "mitigation", "defense", "guardrail", "detection", "method",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def _content_words(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in _STOPWORDS and len(token) > 2]


def _safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:max_len].strip("-") or "gap")


def _normalize_title(title: str) -> str:
    return " ".join(_tokens(title))


def _dedupe_key(paper: dict) -> str:
    doi = (paper.get("doi") or "").lower()
    if doi:
        return f"doi:{doi}"
    arxiv_id = paper.get("arxiv_id") or (
        paper.get("paper_id") if paper.get("source") == "arxiv" else ""
    )
    if arxiv_id:
        return f"arxiv:{str(arxiv_id).lower()}"
    semantic_scholar_id = paper.get("semantic_scholar_id") or (
        paper.get("paper_id") if paper.get("source") == "semantic_scholar" else ""
    )
    if semantic_scholar_id:
        return f"s2:{str(semantic_scholar_id).lower()}"
    return f"title:{_normalize_title(paper.get('title', ''))}"


def _matching_papers(left: list[dict], right: list[dict]) -> bool:
    left_keys = sorted((p.get("source"), p.get("paper_id")) for p in left)
    right_keys = sorted((p.get("source"), p.get("paper_id")) for p in right)
    return left_keys == right_keys


def find_existing_gap_analysis_for_project(project: str) -> dict | None:
    """Return the newest gap analysis artifact matching a project manifest."""
    project_papers = get_project_papers(project)
    for path in sorted(_ANALYSIS_DIR.glob("gap_analysis_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if _matching_papers(data.get("papers", []), project_papers):
                return {
                    "analysis": data.get("analysis", {}),
                    "path": str(path),
                    "papers": data.get("papers", []),
                }
        except Exception:
            logger.warning("Could not inspect gap analysis artifact: %s", path)
    return None


def _gap_entries(gap_analysis: dict) -> list[dict]:
    entries = []
    for index, gap in enumerate(gap_analysis.get("research_gaps") or [], start=1):
        text = gap.get("gap") or ""
        if text:
            entries.append({
                "gap_id": f"research_gap_{index}",
                "gap_type": "research_gap",
                "gap": text,
                "source_gap": gap,
            })
    for index, gap in enumerate(gap_analysis.get("methodological_gaps") or [], start=1):
        text = gap.get("gap") or ""
        if text:
            entries.append({
                "gap_id": f"methodological_gap_{index}",
                "gap_type": "methodological_gap",
                "gap": text,
                "source_gap": gap,
            })
    return entries


def gap_use_for_experiments(status: str, refined_gap: str | None = "") -> bool:
    refined = bool((refined_gap or "").strip())
    if status == "confirmed_candidate_gap":
        return True
    if status == "partially_addressed":
        return True
    if status == "needs_refinement":
        return refined
    if status == "too_broad":
        return refined
    return False


def generate_gap_validation_queries(gap: str, *, max_queries: int = 5) -> list[str]:
    """Generate targeted academic search queries from a candidate gap."""
    words = _content_words(gap)
    counts = Counter(words)
    ranked = [word for word, _ in counts.most_common() if word not in {"real", "world"}]
    domain = [word for word in ranked if word in _DOMAIN_TERMS]
    specific = [word for word in ranked if word not in domain]

    queries = []
    compact_gap = " ".join(words[:8])
    if compact_gap:
        queries.append(compact_gap)

    base_terms = (domain[:4] + specific[:4])[:6]
    if base_terms:
        queries.append(" ".join(base_terms))
        queries.append(" ".join(base_terms[:4] + ["evaluation"]))
        queries.append(" ".join(base_terms[:4] + ["benchmark"]))

    if {"real", "world"}.issubset(set(words)) or "real-world" in words:
        queries.append(" ".join((domain[:4] or ranked[:4]) + ["real-world evaluation"]))
    if "deployment" in words or "production" in words:
        queries.append(" ".join((domain[:4] or ranked[:4]) + ["deployment production evaluation"]))
    if "prompt" in words or "injection" in words:
        queries.append("indirect prompt injection defense real-world agents")

    unique = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique[:max_queries]


def classify_validation_paper(gap: str, paper: dict) -> dict:
    """Classify a search result using only title, abstract, and metadata."""
    gap_words = set(_content_words(gap))
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    text_words = set(_content_words(text))
    if not text_words:
        return {
            "classification": "irrelevant",
            "reason": "No title or abstract text was available for metadata-only assessment.",
        }

    overlap = gap_words & text_words
    overlap_ratio = len(overlap) / max(len(gap_words), 1)
    has_direct_term = bool(text_words & _DIRECT_TERMS)
    has_partial_term = bool(text_words & _PARTIAL_TERMS)

    if overlap_ratio >= 0.45 and has_direct_term:
        classification = "directly_addresses_gap"
        reason = "The metadata overlaps strongly with the gap and describes direct evaluation, deployment, benchmarking, or empirical study."
    elif overlap_ratio >= 0.35 or (overlap_ratio >= 0.20 and has_partial_term):
        classification = "partially_addresses_gap"
        reason = "The metadata covers central gap concepts but appears to address only part of the missing scope."
    elif overlap_ratio >= 0.12:
        classification = "related_but_not_addressing"
        reason = "The paper is topically related, but the metadata does not show that it addresses the gap."
    else:
        classification = "irrelevant"
        reason = "The metadata has little overlap with the candidate gap."

    return {"classification": classification, "reason": reason}


def _normalize_validation_paper(paper: dict, gap: str, project_ids: set[str]) -> dict:
    classification = classify_validation_paper(gap, paper)
    paper_id = paper.get("paper_id") or paper.get("semantic_scholar_id") or paper.get("arxiv_id") or ""
    source = paper.get("source") or ""
    url = paper.get("url") or paper.get("semantic_scholar_url") or ""
    already_in_project = paper_id in project_ids
    return {
        "paper_id": paper_id,
        "source": source,
        "title": paper.get("title") or "",
        "year": paper.get("year") or "",
        "authors": paper.get("authors") or [],
        "doi": paper.get("doi") or "",
        "url": url,
        "pdf_url": paper.get("pdf_url") or "",
        "classification": classification["classification"],
        "reason": classification["reason"],
        "already_in_project": already_in_project,
        "external_to_project": bool(project_ids) and not already_in_project,
    }


def _search_gap_queries(queries: list[str], per_query_limit: int) -> list[dict]:
    results = []
    for query in queries:
        try:
            logger.info("Gap validation search: query=%r limit=%d", query, per_query_limit)
            results.extend(fetch_papers(query, per_query_limit))
        except Exception as e:
            logger.warning("Gap validation search failed for query=%r: %s", query, e)

    deduped = {}
    for paper in results:
        key = _dedupe_key(paper)
        if not key.endswith(":") and key not in deduped:
            deduped[key] = paper
    return list(deduped.values())


def _decision(
    gap: str,
    validation_papers: list[dict],
    project_ids: set[str],
) -> tuple[str, str, str, str, str]:
    external = [p for p in validation_papers if p["paper_id"] not in project_ids]
    direct = [p for p in external if p["classification"] == "directly_addresses_gap"]
    partial = [p for p in external if p["classification"] == "partially_addresses_gap"]
    related = [p for p in external if p["classification"] == "related_but_not_addressing"]
    useful_count = len(direct) + len(partial) + len(related)

    if len(_content_words(gap)) < 4:
        return (
            "too_broad",
            "low",
            "The gap is too short or general to validate reliably with targeted academic search.",
            f"{gap.strip()} in a specific task, environment, population, or evaluation setting",
            "Narrow the gap before running validation again.",
        )
    if len(validation_papers) < 2 or useful_count == 0:
        return (
            "insufficient_evidence",
            "low",
            "The follow-up search returned too little useful metadata to support a strong validation decision.",
            "",
            "Try a more specific gap statement or increase max_results.",
        )
    if len(direct) >= 2:
        return (
            "already_addressed",
            "high",
            "Multiple external papers appear to directly address the candidate gap.",
            "",
            "Review the directly addressing papers and reformulate the gap around what remains missing.",
        )
    if len(direct) == 1 or len(partial) >= 2:
        refined = (
            f"{gap.strip()} with emphasis on settings, benchmarks, or populations not covered by the validation papers"
        )
        return (
            "partially_addressed",
            "medium",
            "Some external papers address part of the gap, but the metadata does not show full coverage of the original claim.",
            refined,
            "Refine the gap and inspect the partially/directly addressing papers before using it as a research claim.",
        )
    if len(related) >= 3:
        return (
            "needs_refinement",
            "medium",
            "Several papers are related, but the current gap wording does not clearly separate the missing contribution from adjacent work.",
            f"{gap.strip()} in a narrower operational or methodological setting",
            "Rewrite the gap with a sharper scope and validate again.",
        )
    return (
        "confirmed_candidate_gap",
        "medium",
        "The search found related literature but little evidence that external papers directly address the candidate gap.",
        "",
        "Use this as a candidate gap, but cite the validation scope and review top related papers manually.",
    )


def _save_validation_artifact(result: dict) -> str:
    _VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _VALIDATION_DIR / f"gap_validation_{timestamp}_{_safe_slug(result['gap'])}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)


def _external_evidence_titles(validation_papers: list[dict], limit: int = 3) -> list[str]:
    useful = {
        "directly_addresses_gap",
        "partially_addresses_gap",
        "related_but_not_addressing",
    }
    titles = []
    for paper in validation_papers:
        if paper.get("already_in_project"):
            continue
        if paper.get("classification") not in useful:
            continue
        title = paper.get("title")
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _compact_validation_result(entry: dict, result: dict) -> dict:
    refined_gap = result.get("refined_gap") or ""
    return {
        "gap_id": entry["gap_id"],
        "gap_type": entry["gap_type"],
        "original_gap": entry["gap"],
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "use_for_experiments": gap_use_for_experiments(result.get("status", ""), refined_gap),
        "decision_reason": result.get("decision_reason", ""),
        "refined_gap": refined_gap or None,
        "artifact_path": result.get("artifact_path", ""),
        "external_evidence_titles": _external_evidence_titles(
            result.get("validation_papers") or [],
            limit=3,
        ),
    }


def _save_batch_validation_summary(summary: dict) -> str:
    _BATCH_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_slug = _safe_slug(summary["project"], max_len=60)
    path = _BATCH_VALIDATION_DIR / f"batch_gap_validation_{timestamp}_{project_slug}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(path)


def batch_validate_gaps(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
    progress_callback=None,
    cancel_check=None,
) -> dict:
    """Validate all research and methodological gaps from a project's cached gap analysis."""
    if not project:
        raise ValueError("batch_validate_gaps requires a project name.")
    if max_results_per_gap < 1:
        raise ValueError("max_results_per_gap must be at least 1.")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if mode != "metadata_only":
        raise ValueError("batch_validate_gaps currently supports only mode='metadata_only'.")

    project_papers = get_project_papers(project)
    cached_gap_analysis = find_existing_gap_analysis_for_project(project)
    if cached_gap_analysis is None:
        raise FileNotFoundError(
            f"No cached gap analysis found for project {project!r}. Run detect_gaps_tool(project=...) first."
        )

    gap_entries = _gap_entries(cached_gap_analysis["analysis"])
    validated_gaps = []
    failed_gaps = []

    logger.info(
        "Batch gap validation started: project=%r gaps=%d max_workers=%d",
        project,
        len(gap_entries),
        min(max_workers, max(len(gap_entries), 1)),
    )

    def _run(entry: dict) -> dict:
        if cancel_check and cancel_check():
            raise RuntimeError("cancel_requested")
        result = validate_gap(
            gap=entry["gap"],
            project=project,
            max_results=max_results_per_gap,
            mode=mode,
        )
        return _compact_validation_result(entry, result)

    with ThreadPoolExecutor(max_workers=min(max_workers, max(len(gap_entries), 1))) as pool:
        futures = {pool.submit(_run, entry): entry for entry in gap_entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                compact_result = future.result()
                validated_gaps.append(compact_result)
                if progress_callback:
                    progress_callback("completed", entry, compact_result, None)
            except Exception as e:
                if str(e) == "cancel_requested":
                    logger.info(
                        "Gap validation cancelled: project=%r gap_id=%s",
                        project,
                        entry["gap_id"],
                    )
                    failed_gaps.append({
                        "gap_id": entry["gap_id"],
                        "gap_type": entry["gap_type"],
                        "original_gap": entry["gap"],
                        "error": "cancel_requested",
                    })
                    if progress_callback:
                        progress_callback("cancelled", entry, None, "cancel_requested")
                    continue
                logger.error(
                    "Gap validation failed: project=%r gap_id=%s error=%s",
                    project,
                    entry["gap_id"],
                    e,
                )
                failed_gaps.append({
                    "gap_id": entry["gap_id"],
                    "gap_type": entry["gap_type"],
                    "original_gap": entry["gap"],
                    "error": str(e),
                })
                if progress_callback:
                    progress_callback("failed", entry, None, str(e))

    order = {entry["gap_id"]: index for index, entry in enumerate(gap_entries)}
    validated_gaps.sort(key=lambda item: order.get(item["gap_id"], 999))
    failed_gaps.sort(key=lambda item: order.get(item["gap_id"], 999))

    status_counts = {}
    for item in validated_gaps:
        status = item.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "project": project,
        "project_papers": project_papers,
        "gap_analysis_path": cached_gap_analysis["path"],
        "gap_count": len(gap_entries),
        "validated_count": len(validated_gaps),
        "failed_count": len(failed_gaps),
        "mode": mode,
        "max_results_per_gap": max_results_per_gap,
        "status_counts": status_counts,
        "validated_gaps": validated_gaps,
        "failed_gaps": failed_gaps,
        "batch_artifact_path": "",
    }
    summary["batch_artifact_path"] = _save_batch_validation_summary(summary)
    logger.info(
        "Batch gap validation complete: project=%r validated=%d failed=%d artifact=%s",
        project,
        summary["validated_count"],
        summary["failed_count"],
        summary["batch_artifact_path"],
    )
    return summary


def find_latest_batch_validation_summary(project: str, papers: list[dict] | None = None) -> dict | None:
    """Return the newest batch validation summary for a project and optional paper set."""
    if not project:
        return None
    for path in sorted(_BATCH_VALIDATION_DIR.glob("batch_gap_validation_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("project") != project:
                continue
            if papers is not None and not _matching_papers(data.get("project_papers", []), papers):
                continue
            data["batch_artifact_path"] = data.get("batch_artifact_path") or str(path)
            return data
        except Exception:
            logger.warning("Could not inspect batch validation artifact: %s", path)
    return None


def validated_gap_analysis_for_experiments(summary: dict) -> tuple[dict, dict]:
    """Build compact gap-analysis input and metadata from a batch validation summary."""
    included_research = []
    included_methodological = []
    excluded = []
    refined_count = 0

    for item in summary.get("validated_gaps") or []:
        original_gap = item.get("original_gap", "")
        refined_gap = item.get("refined_gap") or ""
        use_for_experiments = bool(item.get("use_for_experiments"))
        if use_for_experiments:
            gap_text = refined_gap or original_gap
            if refined_gap:
                refined_count += 1
            gap_payload = {
                "gap": gap_text,
                "original_gap": original_gap,
                "validation_status": item.get("status"),
                "validation_confidence": item.get("confidence"),
                "validation_decision_reason": item.get("decision_reason", ""),
                "external_evidence_titles": item.get("external_evidence_titles", [])[:3],
                "relevant_papers": [],
            }
            if item.get("gap_type") == "methodological_gap":
                gap_payload["current_approaches"] = []
                gap_payload["missing_approach"] = gap_text
                included_methodological.append(gap_payload)
            else:
                gap_payload["evidence"] = item.get("decision_reason", "")
                included_research.append(gap_payload)
        else:
            excluded.append({
                "gap": original_gap,
                "status": item.get("status"),
                "reason": _exclusion_reason(item),
            })

    gap_analysis = {
        "research_gaps": included_research,
        "methodological_gaps": included_methodological,
        "contradictions": [],
        "connections": [],
        "field_summary": (
            "Experiment suggestions are based on gaps retained after batch validation."
        ),
    }
    metadata = {
        "validation_used": True,
        "batch_validation_path": summary.get("batch_artifact_path"),
        "included_gap_count": len(included_research) + len(included_methodological),
        "excluded_gap_count": len(excluded),
        "excluded_gaps": excluded,
        "refined_gap_count": refined_count,
    }
    return gap_analysis, metadata


def _exclusion_reason(item: dict) -> str:
    status = item.get("status")
    if status == "already_addressed":
        return "Skipped because validation found direct external evidence."
    if status == "too_broad":
        return "Skipped because validation judged the gap too broad and no refined gap was available."
    if status == "insufficient_evidence":
        return "Skipped because validation found insufficient useful evidence."
    if status == "needs_refinement":
        return "Skipped because validation requires refinement and no refined gap was available."
    return "Skipped because validation did not mark the gap for experiment generation."


def validate_gap(
    gap: str,
    project: str | None = None,
    max_results: int = 10,
    mode: str = "metadata_only",
) -> dict:
    """
    Validate a candidate gap against wider academic metadata search results.
    """
    if not gap or not gap.strip():
        raise ValueError("validate_gap requires a non-empty gap.")
    if mode != "metadata_only":
        raise ValueError("validate_gap currently supports only mode='metadata_only'.")
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    project_papers = get_project_papers(project) if project else []
    project_ids = {paper.get("paper_id") for paper in project_papers if paper.get("paper_id")}
    search_queries = generate_gap_validation_queries(gap)
    per_query_limit = max(1, min(max_results, 10))
    search_results = _search_gap_queries(search_queries, per_query_limit)
    validation_papers = [
        _normalize_validation_paper(paper, gap, project_ids)
        for paper in search_results[:max_results]
    ]

    relevant_classes = {
        "directly_addresses_gap",
        "partially_addresses_gap",
        "related_but_not_addressing",
    }
    relevant_results = sum(
        1 for paper in validation_papers if paper["classification"] in relevant_classes
    )
    status, confidence, decision_reason, refined_gap, recommended_next_step = _decision(
        gap,
        validation_papers,
        project_ids,
    )

    result = {
        "gap": gap,
        "project": project,
        "mode": mode,
        "search_queries": search_queries,
        "results_found": len(validation_papers),
        "relevant_results": relevant_results,
        "status": status,
        "confidence": confidence,
        "decision_reason": decision_reason,
        "refined_gap": refined_gap,
        "validation_papers": validation_papers,
        "recommended_next_step": recommended_next_step,
        "artifact_path": "",
    }
    result["artifact_path"] = _save_validation_artifact(result)
    logger.info(
        "Gap validation complete: status=%s confidence=%s results=%d relevant=%d artifact=%s",
        status,
        confidence,
        result["results_found"],
        relevant_results,
        result["artifact_path"],
    )
    return result
