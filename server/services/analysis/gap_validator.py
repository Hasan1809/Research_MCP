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
_SECURITY_TERMS = {
    "attack", "attacks", "adversarial", "backdoor", "compromise", "defense",
    "defenses", "guardrail", "injection", "jailbreak", "malicious",
    "mitigation", "privacy", "risk", "risks", "safety", "secure",
    "security", "threat", "threats", "trust", "trustworthiness",
    "vulnerabilities", "vulnerability",
}
_GENERAL_SECURITY_DOMAINS = {
    "android", "automotive", "binary", "connected", "cve", "firmware",
    "fuzzer", "fuzzing", "iot", "kernel", "linux", "malware", "network",
    "networks", "operating", "robot", "robotics", "software", "system",
    "systems", "vehicle", "vehicles", "vanet", "web",
}
_BLOCKING_NEGATIVE_PATTERNS = [
    r"\brobot(?:ic|ics)? policy evaluation\b",
    r"\bdriver (?:movement|modeling|behaviou?r)\b",
    r"\bconnected vehicle[s]?\b",
    r"\bcyber[- ]physical system[s]?\b",
    r"\bindustrial control system[s]?\b",
    r"\bgeneric hci\b",
    r"\bgui usability\b",
    r"\bprocess modeling copilot[s]?\b",
]
_AGENT_RELEVANCE_PATTERNS = [
    r"\bllm[- ]?based agent[s]?\b",
    r"\bllm agent[s]?\b",
    r"\bllm[- ]?integrated (?:app|application) system[s]?\b",
    r"\bllm[- ]?integrated application[s]?\b",
    r"\bllm[- ]?powered gui agent[s]?\b",
    r"\blanguage model agent[s]?\b",
    r"\blarge language model agent[s]?\b",
    r"\bagentic ai\b",
    r"\bagentic app system[s]?\b",
    r"\bagentic system[s]?\b",
    r"\bagent workflow[s]?\b",
    r"\btool[- ]?(?:using|use) (?:llm|language model|agent)[s]?\b",
    r"\btool orchestration\b",
    r"\b(?:llm|language model)[s]? (?:with|using) tool[s]?\b",
    r"\bautonomous agent[s]?\b",
    r"\bautonomous ai agent[s]?\b",
    r"\bweb agent[s]?\b",
    r"\bweb[- ]use agent[s]?\b",
    r"\bcomputer[- ]use agent[s]?\b",
    r"\bmulti[- ]agent llm[s]?\b",
    r"\bmulti[- ]agent llm system[s]?\b",
    r"\bmcp\b",
    r"\bfunction calling\b",
]
_LLM_AGENT_PROJECT_TERMS = {
    "agent", "agents", "agentic", "llm", "llms", "llm-agent", "mcp", "tool-use",
    "tool", "tools", "computer-use", "web-use",
}
_GENERIC_REFINEMENT_FACETS = {
    "LLM agents", "security", "evaluation", "benchmark", "metric",
    "framework", "real-world",
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
    paths = list(_ANALYSIS_DIR.glob("gap_analysis_*.json")) + list(_ANALYSIS_DIR.glob("lc_gap_analysis_*.json"))
    for path in sorted(paths, reverse=True):
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


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _project_context_is_llm_agent_security(project_context: str | None) -> bool:
    words = set(_tokens(project_context or ""))
    return bool(words & _LLM_AGENT_PROJECT_TERMS) and bool(words & _SECURITY_TERMS)


def generate_gap_validation_queries(
    gap: str,
    *,
    max_queries: int = 5,
    project_context: str | None = None,
) -> list[str]:
    """Generate targeted academic search queries from a candidate gap."""
    facets = extract_gap_facets(gap, project_context=project_context)
    words = _content_words(gap)
    counts = Counter(words)
    ranked = [word for word, _ in counts.most_common() if word not in {"real", "world"}]
    domain = [word for word in ranked if word in _DOMAIN_TERMS]
    specific = [word for word in ranked if word not in domain]

    queries = []
    gap_terms = " ".join((specific or ranked)[:5])
    anchors = [
        "LLM agent security",
        "LLM-based agent security",
        "tool-use LLM agent security",
        "LLM agent prompt injection",
        "LLM agent tool use vulnerabilities",
        "LLM-integrated app security",
        "agentic AI security",
    ]

    if "real-world deployment" in facets.get("contribution", []) or "deployment" in facets.get("setting", []):
        queries.append(f"LLM agent security real-world deployment evaluation {gap_terms}".strip())
    if "standardized metrics" in facets.get("contribution", []):
        queries.append("LLM agent security standardized evaluation metrics benchmark")
    if "explainability" in facets.get("contribution", []):
        queries.append("LLM agent security explainability interpretability unsafe tool choices")
    if "human-centered evaluation" in facets.get("contribution", []):
        queries.append("human-centered evaluation LLM agent security users tool use")
    if "prompt injection" in facets.get("threat_model", []):
        queries.append("LLM agent prompt injection tool use vulnerabilities")

    if "LLM agents" in facets.get("domain", []) or _project_context_is_llm_agent_security(project_context):
        for anchor in anchors:
            if gap_terms:
                queries.append(f"{anchor} {gap_terms}")
            else:
                queries.append(anchor)
    else:
        compact_gap = " ".join(words[:8])
        if compact_gap:
            queries.append(compact_gap)
        base_terms = (domain[:4] + specific[:4])[:6]
        if base_terms:
            queries.append(" ".join(base_terms))

    unique = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique[:max_queries]


def extract_gap_facets(gap: str, project_context: str | None = None) -> dict:
    """Extract coarse validation facets from a gap without an extra LLM call."""
    text = (gap or "").lower()
    context = (project_context or "").lower()
    combined = f"{text} {context}"
    words = set(_tokens(text))
    combined_words = set(_tokens(combined))
    facets = {
        "domain": [],
        "security_topic": [],
        "contribution": [],
        "threat_model": [],
        "specific_problem": [],
        "setting": [],
        "method_type": [],
        "population": [],
        "evaluation_target": [],
    }

    if _has_llm_agent_relevance(combined) or _project_context_is_llm_agent_security(project_context):
        _append_unique(facets["domain"], "LLM agents")
    if any(phrase in combined for phrase in ("llm-integrated app", "llm-integrated application")):
        _append_unique(facets["domain"], "LLM-integrated app systems")
    if "multi-agent" in combined or "multi agent" in combined:
        _append_unique(facets["domain"], "multi-agent LLM systems")

    if any(term in combined_words for term in _SECURITY_TERMS):
        _append_unique(facets["security_topic"], "security")
    if "tool" in words or "tools" in words or "tool-use" in text or "tool use" in text:
        _append_unique(facets["security_topic"], "tool-use security")
        _append_unique(facets["setting"], "tool-use workflow")
    if "prompt" in words or "injection" in words or "adversarial prompting" in text:
        _append_unique(facets["security_topic"], "prompt injection")
        _append_unique(facets["threat_model"], "prompt injection")
    if "indirect prompt injection" in text:
        _append_unique(facets["threat_model"], "indirect prompt injection")
    if "vulnerabilities" in words or "vulnerability" in words:
        _append_unique(facets["security_topic"], "vulnerability analysis")
    if "defense" in words or "defenses" in words or "guardrail" in words:
        _append_unique(facets["security_topic"], "defense")
    for phrase, facet in (
        ("tool misuse", "tool misuse"),
        ("unsafe action", "unsafe action"),
        ("sandbox", "sandbox"),
        ("access control", "access control"),
        ("policy bypass", "policy bypass"),
        ("privacy", "privacy"),
        ("malicious instruction", "malicious instruction"),
        ("adversarial prompting", "adversarial prompting"),
    ):
        if phrase in text:
            _append_unique(facets["threat_model"], facet)

    if "real-world" in text or {"real", "world"}.issubset(words):
        _append_unique(facets["setting"], "real-world")
        _append_unique(facets["contribution"], "real-world deployment")
    if "dynamic" in words or "dynamic environment" in text:
        _append_unique(facets["setting"], "dynamic environment")
        _append_unique(facets["contribution"], "dynamic environment")
    if "deployment" in words or "deployed" in words or "production" in words:
        _append_unique(facets["setting"], "deployment")
        _append_unique(facets["contribution"], "real-world deployment")
    if "multi-agent" in text or {"multi", "agent"}.issubset(words):
        _append_unique(facets["setting"], "multi-agent")
        _append_unique(facets["contribution"], "multi-agent setting")

    if "human-centered" in text or "human" in words:
        _append_unique(facets["method_type"], "human-centered")
        _append_unique(facets["contribution"], "human-centered evaluation")
    if "user" in words or "users" in words or "participant" in words or "participants" in words:
        _append_unique(facets["population"], "users")
        _append_unique(facets["contribution"], "user study")
    if "non-technical" in text or "nontechnical" in words:
        _append_unique(facets["population"], "non-technical users")
    if "machine" in words and "learning" in words:
        _append_unique(facets["method_type"], "machine-learning")
    if "simulation" in words or "simulation-based" in text:
        _append_unique(facets["method_type"], "simulation-based")
        _append_unique(facets["contribution"], "simulation-based evaluation")
    if "explainability" in words or "interpretability" in words or "interpretable" in words:
        _append_unique(facets["method_type"], "explainability")
        _append_unique(facets["contribution"], "explainability")
    if "detect" in words or "detection" in words:
        _append_unique(facets["contribution"], "vulnerability detection")
    if "mitigate" in words or "mitigation" in words:
        _append_unique(facets["contribution"], "mitigation")

    if "benchmark" in words or "benchmarks" in words:
        _append_unique(facets["evaluation_target"], "benchmark")
        _append_unique(facets["contribution"], "benchmark")
    if "metric" in words or "metrics" in words:
        _append_unique(facets["evaluation_target"], "metric")
        _append_unique(facets["contribution"], "standardized metrics")
    if "standardized" in words or "standard" in words:
        _append_unique(facets["contribution"], "standardized metrics")
    if "framework" in words:
        _append_unique(facets["contribution"], "evaluation framework")
    if "evaluation" in words or "evaluate" in words or "evaluating" in words:
        _append_unique(facets["evaluation_target"], "evaluation")

    specific = [word for word in _content_words(gap) if word not in _DOMAIN_TERMS]
    facets["specific_problem"] = specific[:8]
    return facets


def _has_llm_agent_relevance(text: str) -> bool:
    normalized = (text or "").lower()
    if any(re.search(pattern, normalized) for pattern in _AGENT_RELEVANCE_PATTERNS):
        return True
    words = set(_tokens(normalized))
    has_llm = bool({"llm", "llms"} & words) or "language model" in normalized
    has_agent = bool({"agent", "agents", "agentic"} & words)
    has_tool_context = bool({"tool", "tools"} & words) or "function calling" in normalized
    return has_llm and (has_agent or has_tool_context)


def _has_security_relevance(text: str) -> bool:
    return bool(set(_tokens(text)) & _SECURITY_TERMS)


def _is_general_security_without_agent_context(text: str) -> bool:
    words = set(_tokens(text))
    return bool(words & _GENERAL_SECURITY_DOMAINS) and not _has_llm_agent_relevance(text)


def _blocking_negative_reason(text: str) -> str:
    normalized = (text or "").lower()
    if _has_llm_agent_relevance(normalized) and _has_security_relevance(normalized):
        return ""
    for pattern in _BLOCKING_NEGATIVE_PATTERNS:
        if re.search(pattern, normalized):
            return "The paper is primarily about an adjacent non-LLM-agent-security domain."
    if _is_general_security_without_agent_context(normalized):
        return "The paper appears to be general cybersecurity evidence without clear LLM-agent context."
    if "hci" in _tokens(normalized) and not _has_security_relevance(normalized):
        return "The paper appears to be generic HCI without LLM-agent-security context."
    return ""


def _facet_label(key: str, value: str) -> str:
    labels = {
        "domain": value,
        "security_topic": value,
        "contribution": value,
        "threat_model": value,
        "setting": value,
        "method_type": value,
        "population": value,
        "evaluation_target": value,
    }
    return labels.get(key, value)


def _paper_relevance_flags(text: str) -> dict:
    blocking_reason = _blocking_negative_reason(text)
    return {
        "llm_agent_relevant": _has_llm_agent_relevance(text),
        "security_relevant": _has_security_relevance(text),
        "general_security_without_agent_context": _is_general_security_without_agent_context(text),
        "blocking_negative_reason": blocking_reason,
        "has_blocking_negative": bool(blocking_reason),
        "survey_or_benchmark": bool(set(_tokens(text)) & {"survey", "benchmark", "benchmarks", "dataset", "taxonomy"}),
        "direct_study_language": bool(set(_tokens(text)) & _DIRECT_TERMS),
    }


def _text_matches_facet(text: str, key: str, value: str) -> bool:
    normalized = (text or "").lower()
    words = set(_tokens(normalized))
    negates_user_study = any(
        phrase in normalized
        for phrase in (
            "no user study",
            "without user",
            "without users",
            "do not run a user",
            "does not run a user",
            "not run a user",
        )
    )
    if key == "domain" and value == "LLM agents":
        return _has_llm_agent_relevance(normalized)
    if key == "domain" and value == "LLM-integrated app systems":
        return "llm-integrated" in normalized or ("llm" in words and bool({"app", "apps", "application", "applications", "system", "systems"} & words))
    if key == "domain" and value == "multi-agent LLM systems":
        return ("multi-agent" in normalized or {"multi", "agent"}.issubset(words)) and ("llm" in words or "language model" in normalized)
    if key == "security_topic":
        if value == "security":
            return _has_security_relevance(normalized)
        if value == "tool-use security":
            return bool({"tool", "tools", "function", "mcp"} & words) or "tool use" in normalized or "tool-use" in normalized
        if value == "prompt injection":
            return "prompt injection" in normalized or "indirect prompt" in normalized or "injection" in words
        if value == "vulnerability analysis":
            return bool({"vulnerability", "vulnerabilities", "risk", "risks"} & words)
        if value == "defense":
            return bool({"defense", "defenses", "guardrail", "mitigation", "protection"} & words)
    if key == "setting":
        if value == "real-world":
            return "real-world" in normalized or {"real", "world"}.issubset(words) or bool({"field", "realistic"} & words)
        if value == "deployment":
            return bool({"deployment", "deployed", "production"} & words)
        if value == "multi-agent":
            return "multi-agent" in normalized or {"multi", "agent"}.issubset(words) or "distributed agent" in normalized
        if value == "tool-use setting":
            return bool({"tool", "tools", "function", "mcp"} & words) or "tool use" in normalized or "tool-use" in normalized
        if value == "tool-use workflow":
            return bool({"tool", "tools", "function", "mcp", "workflow", "workflows"} & words) or "tool use" in normalized or "tool-use" in normalized
        if value == "dynamic environment":
            return bool({"dynamic", "changing", "adaptive", "deployment", "production"} & words)
    if key == "method_type":
        if value == "human-centered":
            if negates_user_study:
                return False
            return "human-centered" in normalized or bool({"human", "user", "users", "participant", "participants", "usability"} & words)
        if value == "machine-learning":
            return "machine learning" in normalized or bool({"classifier", "classifiers", "learning"} & words)
        if value == "simulation-based":
            return "simulation" in words or "simulation-based" in normalized or "simulated" in words
        if value == "explainability":
            return bool({"explainability", "explainable", "interpretability", "interpretable", "explanation", "attribution"} & words)
    if key == "population":
        if value == "users":
            if negates_user_study:
                return False
            return bool({"user", "users", "participant", "participants", "human", "humans"} & words)
        if value == "non-technical users":
            return "non-technical" in normalized or "nontechnical" in words
    if key == "evaluation_target":
        if value == "benchmark":
            return bool({"benchmark", "benchmarks", "dataset", "suite"} & words)
        if value == "metric":
            return bool({"metric", "metrics", "measure", "measurement"} & words)
        if value == "evaluation":
            return bool({"evaluate", "evaluates", "evaluating", "evaluation", "study", "studies", "experiment", "experiments"} & words)
    if key == "threat_model":
        if value == "prompt injection":
            return "prompt injection" in normalized or "indirect prompt" in normalized or bool({"prompt", "injection"} & words)
        if value == "indirect prompt injection":
            return "indirect prompt injection" in normalized
        if value == "tool misuse":
            return "tool misuse" in normalized or ("tool" in words and bool({"misuse", "unsafe", "malicious"} & words))
        if value == "unsafe action":
            return "unsafe action" in normalized or bool({"unsafe", "risky"} & words)
        if value in {"sandbox", "access control", "policy bypass", "privacy", "malicious instruction", "adversarial prompting"}:
            return value in normalized
    if key == "contribution":
        if value == "real-world deployment":
            return "real-world" in normalized or bool({"deployment", "deployed", "production", "field", "realistic"} & words)
        if value == "dynamic environment":
            return bool({"dynamic", "changing", "adaptive"} & words)
        if value == "benchmark":
            return bool({"benchmark", "benchmarks", "dataset", "suite"} & words)
        if value == "standardized metrics":
            return bool({"standardized", "standard", "metric", "metrics", "measurement"} & words)
        if value == "evaluation framework":
            return "framework" in words and bool({"evaluate", "evaluation", "benchmark", "metric", "metrics"} & words)
        if value == "explainability":
            return bool({"explainability", "explainable", "interpretability", "interpretable", "explanation", "attribution"} & words)
        if value == "human-centered evaluation":
            if negates_user_study:
                return False
            return "human-centered" in normalized or bool({"human", "user", "users", "participant", "participants"} & words)
        if value == "user study":
            if negates_user_study:
                return False
            return bool({"user", "users", "participant", "participants", "study", "survey"} & words)
        if value == "multi-agent setting":
            return "multi-agent" in normalized or {"multi", "agent"}.issubset(words)
        if value == "tool-use workflow":
            return "tool use" in normalized or "tool-use" in normalized or bool({"tool", "tools", "workflow", "workflows"} & words)
        if value == "simulation-based evaluation":
            return bool({"simulation", "simulated", "evaluate", "evaluation"} & words)
        if value == "vulnerability detection":
            return bool({"detect", "detection", "detector", "vulnerability", "vulnerabilities"} & words)
        if value == "mitigation":
            return bool({"mitigate", "mitigation", "defense", "defenses", "guardrail"} & words)
    return False


def _score_facet_match(facets: dict, text: str) -> tuple[float, list[str], list[str]]:
    matched = []
    missing = []
    weighted_total = 0.0
    weighted_score = 0.0
    weights = {
        "domain": 2.5,
        "security_topic": 2.0,
        "contribution": 2.0,
        "threat_model": 1.75,
        "setting": 1.25,
        "method_type": 1.25,
        "population": 1.25,
        "evaluation_target": 1.0,
    }

    for key, values in facets.items():
        if key == "specific_problem":
            continue
        for value in values or []:
            weight = weights.get(key, 1.0)
            weighted_total += weight
            label = _facet_label(key, value)
            if _text_matches_facet(text, key, value):
                matched.append(label)
                weighted_score += weight
            else:
                missing.append(label)

    if weighted_total == 0:
        return 0.0, matched, missing
    return round(weighted_score / weighted_total, 2), matched, missing


def classify_validation_paper(
    gap: str,
    paper: dict,
    *,
    project_context: str | None = None,
) -> dict:
    """Classify a search result using only title, abstract, and metadata."""
    gap_words = set(_content_words(gap))
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    text_words = set(_content_words(text))
    facets = extract_gap_facets(gap, project_context=project_context)
    if not text_words:
        return {
            "classification": "irrelevant",
            "reason": "No title or abstract text was available for metadata-only assessment.",
            "decision_reason": "No title or abstract text was available for metadata-only assessment.",
            "score": 0.0,
            "evidence_score": 0.0,
            "matched_facets": [],
            "missing_facets": [],
            "matched_domain_facets": [],
            "matched_security_facets": [],
            "matched_contribution_facets": [],
            "missing_core_facets": [],
            "blocking_negative_reason": "",
            "relevance_flags": _paper_relevance_flags(text),
        }

    overlap = gap_words & text_words
    overlap_ratio = len(overlap) / max(len(gap_words), 1)
    has_partial_term = bool(text_words & _PARTIAL_TERMS)
    flags = _paper_relevance_flags(text)
    facet_score, matched_facets, missing_facets = _score_facet_match(facets, text)
    matched_domain = [value for value in facets.get("domain", []) if _text_matches_facet(text, "domain", value)]
    missing_domain = [value for value in facets.get("domain", []) if value not in matched_domain]
    matched_security = [value for value in facets.get("security_topic", []) if _text_matches_facet(text, "security_topic", value)]
    missing_security = [value for value in facets.get("security_topic", []) if value not in matched_security]
    matched_contribution = [value for value in facets.get("contribution", []) if _text_matches_facet(text, "contribution", value)]
    missing_contribution = [value for value in facets.get("contribution", []) if value not in matched_contribution]
    matched_threats = [value for value in facets.get("threat_model", []) if _text_matches_facet(text, "threat_model", value)]
    missing_threats = [value for value in facets.get("threat_model", []) if value not in matched_threats]
    requires_llm_agent = bool(facets.get("domain")) or _project_context_is_llm_agent_security(project_context)
    requires_security = bool(facets.get("security_topic"))
    has_contribution_requirement = bool(facets.get("contribution"))
    core_missing = []
    if requires_llm_agent and not matched_domain:
        core_missing.append("LLM-agent domain")
    if requires_security and not matched_security:
        core_missing.append("security context")
    if has_contribution_requirement and not matched_contribution:
        core_missing.append("missing contribution")
    if facets.get("threat_model") and not matched_threats:
        core_missing.append("threat model")

    if requires_llm_agent and not flags["llm_agent_relevant"]:
        has_substantive_adjacent_match = (
            flags["security_relevant"]
            and (matched_security or matched_threats or matched_contribution)
            and facet_score >= 0.45
        )
        classification = "related_but_not_addressing" if has_substantive_adjacent_match else "irrelevant"
        reason = "The paper has adjacent security or keyword overlap, but the metadata does not show clear LLM-agent relevance."
    elif requires_security and not flags["security_relevant"]:
        has_substantive_agent_match = (
            flags["llm_agent_relevant"]
            and (matched_domain or matched_contribution)
            and facet_score >= 0.45
        )
        classification = "related_but_not_addressing" if has_substantive_agent_match else "irrelevant"
        reason = "The paper is near the LLM-agent topic, but the metadata does not show security relevance."
    elif flags["has_blocking_negative"]:
        classification = "related_but_not_addressing"
        reason = flags["blocking_negative_reason"]
    elif (
        facet_score >= 0.82
        and flags["direct_study_language"]
        and not core_missing
        and (not has_contribution_requirement or matched_contribution)
    ):
        classification = "directly_addresses_gap"
        reason = "The metadata matches the LLM-agent security context, security requirements, and missing contribution with direct study or evaluation language."
    elif (
        facet_score >= 0.78
        and flags["direct_study_language"]
        and len(core_missing) == 0
        and len(missing_facets) <= 1
        and (not has_contribution_requirement or matched_contribution)
    ):
        classification = "directly_addresses_gap"
        reason = "The metadata strongly matches the core gap facets and appears to directly study the same issue."
    elif facet_score >= 0.55 or (overlap_ratio >= 0.20 and has_partial_term):
        classification = "partially_addresses_gap"
        if core_missing:
            reason = f"The paper covers nearby concepts but misses core facets: {', '.join(core_missing[:3])}."
        elif missing_facets:
            reason = f"The paper covers the same general topic but misses important facets: {', '.join(missing_facets[:3])}."
        else:
            reason = "The paper covers the same general topic, but the metadata is not strong enough to close the exact gap."
    elif overlap_ratio >= 0.12 or flags["llm_agent_relevant"] or flags["security_relevant"]:
        classification = "related_but_not_addressing"
        reason = "The paper is topically related, but the metadata does not show that it addresses the specific gap."
    else:
        classification = "irrelevant"
        reason = "The metadata has little overlap with the candidate gap."

    return {
        "classification": classification,
        "reason": reason,
        "decision_reason": reason,
        "score": facet_score,
        "evidence_score": facet_score,
        "matched_facets": matched_facets,
        "missing_facets": missing_facets,
        "matched_domain_facets": matched_domain,
        "matched_security_facets": matched_security,
        "matched_contribution_facets": matched_contribution,
        "missing_core_facets": core_missing,
        "blocking_negative_reason": flags["blocking_negative_reason"],
        "relevance_flags": flags,
    }


def _normalize_validation_paper(
    paper: dict,
    gap: str,
    project_ids: set[str],
    project_context: str | None = None,
) -> dict:
    classification = classify_validation_paper(gap, paper, project_context=project_context)
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
        "score": classification.get("score", 0.0),
        "evidence_score": classification.get("evidence_score", classification.get("score", 0.0)),
        "matched_facets": classification.get("matched_facets", []),
        "missing_facets": classification.get("missing_facets", []),
        "matched_domain_facets": classification.get("matched_domain_facets", []),
        "matched_security_facets": classification.get("matched_security_facets", []),
        "matched_contribution_facets": classification.get("matched_contribution_facets", []),
        "missing_core_facets": classification.get("missing_core_facets", []),
        "blocking_negative_reason": classification.get("blocking_negative_reason", ""),
        "decision_reason": classification.get("decision_reason", classification["reason"]),
        "relevance_flags": classification.get("relevance_flags", {}),
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


def _build_refined_gap(gap: str, evidence: list[dict]) -> str:
    """Refine a partially covered gap around facets missing from the evidence."""
    gap_lower = (gap or "").lower()
    if "indirect prompt injection" in gap_lower:
        return (
            "Developing a standardized benchmark for comparing indirect prompt "
            "injection defenses in LLM agents across realistic tool-use workflows."
        )
    if "prompt infection" in gap_lower:
        return (
            "Testing Prompt Infection propagation in multi-agent LLM workflows "
            "with heterogeneous agents, shared memory, and delegated tool use."
        )
    if "subliminal" in gap_lower and ("human" in gap_lower or "collaboration" in gap_lower):
        return (
            "Measuring whether subliminally biased LLM agents alter human trust, "
            "reliance, or decision quality during collaborative multi-agent tasks."
        )
    if "explainability" in gap_lower or "interpretability" in gap_lower:
        return (
            "Evaluating whether explainability methods help developers identify "
            "unsafe tool choices in LLM agents under prompt-injection attacks."
        )
    if "human-centered" in gap_lower or "human" in gap_lower or "user" in gap_lower:
        return (
            "Studying how non-expert users understand and respond to LLM-agent "
            "security warnings during risky tool-use workflows."
        )
    if "real-world" in gap_lower or "deployment" in gap_lower or "dynamic" in gap_lower:
        return (
            "Evaluating LLM-agent security defenses under realistic multi-step "
            "tool-use workflows and deployment constraints."
        )
    if "simulation" in gap_lower:
        return (
            "Testing whether simulation-based environments can reliably predict "
            "LLM-agent security failures in real tool-use deployments."
        )
    if ("machine" in gap_lower and "learning" in gap_lower) or "detect" in gap_lower:
        return (
            "Evaluating learned detectors for subliminal or prompt-injection-driven "
            "unsafe behavior in multi-agent LLM workflows."
        )

    missing_counts = Counter()
    for paper in evidence:
        for facet in paper.get("missing_facets") or []:
            if facet in _GENERIC_REFINEMENT_FACETS:
                continue
            missing_counts[facet] += 1

    missing = [facet for facet, _ in missing_counts.most_common(4)]
    gap_text = gap.strip()
    if not missing:
        return (
            f"{gap_text} in a concrete LLM-agent threat model with explicit "
            "tool-use workflow, evaluation target, and failure criterion."
        )

    additions = []
    if any("human-centered" in facet or "users" in facet for facet in missing):
        additions.append("human-centered evaluation with realistic users")
    if any("real-world" in facet or "deployment" in facet for facet in missing):
        additions.append("real-world deployment settings")
    if any("tool-use" in facet for facet in missing):
        additions.append("realistic tool-use settings")
    if any("multi-agent" in facet for facet in missing):
        additions.append("multi-agent environments")
    if any("benchmark" in facet or "metric" in facet for facet in missing):
        additions.append("explicit benchmark or metric design")

    if not additions:
        additions = [facet for facet in missing[:2] if facet not in _GENERIC_REFINEMENT_FACETS]
    if not additions:
        return (
            f"{gap_text} in a concrete LLM-agent threat model with explicit "
            "tool-use workflow, evaluation target, and failure criterion."
        )
    return f"{gap_text} focused on {', '.join(additions[:3])}."


def _decision(
    gap: str,
    validation_papers: list[dict],
    project_ids: set[str],
) -> tuple[str, str, str, str, str]:
    external = [p for p in validation_papers if p["paper_id"] not in project_ids]
    direct = [p for p in external if p["classification"] == "directly_addresses_gap"]
    strong_direct = [
        p for p in direct
        if float(p.get("evidence_score") or p.get("score") or 0.0) >= 0.82
        and not p.get("missing_core_facets")
        and p.get("matched_contribution_facets")
    ]
    partial = [p for p in external if p["classification"] == "partially_addresses_gap"]
    related = [p for p in external if p["classification"] == "related_but_not_addressing"]
    qualified_related = [
        p for p in related
        if (
            p.get("matched_domain_facets")
            and (p.get("matched_security_facets") or p.get("matched_contribution_facets"))
            and float(p.get("evidence_score") or p.get("score") or 0.0) >= 0.45
        )
        or (
            p.get("matched_security_facets")
            and p.get("matched_contribution_facets")
            and float(p.get("evidence_score") or p.get("score") or 0.0) >= 0.55
        )
    ]
    useful_count = len(direct) + len(partial) + len(qualified_related)
    generic_or_blocked_direct = [
        p for p in strong_direct
        if p.get("blocking_negative_reason") or not p.get("matched_domain_facets")
    ]
    contribution_sets = {
        tuple(sorted(p.get("matched_contribution_facets") or []))
        for p in strong_direct
    }
    coherent_direct_evidence = len(contribution_sets) <= 2

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
    if len(strong_direct) >= 3 and not generic_or_blocked_direct and coherent_direct_evidence:
        return (
            "already_addressed",
            "high",
            "At least three external papers strongly match the LLM-agent security gap facets and directly address the candidate gap.",
            "",
            "Review the directly addressing papers and reformulate the gap around what remains missing.",
        )
    if len(strong_direct) >= 2 and not generic_or_blocked_direct and coherent_direct_evidence:
        return (
            "already_addressed",
            "medium",
            "At least two external papers strongly match the LLM-agent security context and exact missing contribution.",
            "",
            "Review the directly addressing papers and reformulate the gap around what remains missing.",
        )
    if len(strong_direct) >= 1 or len(partial) >= 3:
        refined = _build_refined_gap(gap, direct + partial + related)
        return (
            "partially_addressed",
            "medium",
            "Some external papers address part of the gap, but the metadata does not show full coverage of the original claim.",
            refined,
            "Refine the gap and inspect the partially/directly addressing papers before using it as a research claim.",
        )
    if len(qualified_related) >= 3:
        refined = _build_refined_gap(gap, qualified_related)
        return (
            "needs_refinement",
            "medium",
            "Several papers are related, but the current gap wording does not clearly separate the missing contribution from adjacent work.",
            refined,
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


def _evidence_items(
    validation_papers: list[dict],
    classifications: set[str],
    limit: int = 3,
) -> list[dict]:
    items = []
    seen_titles = set()
    for paper in validation_papers:
        if paper.get("already_in_project"):
            continue
        if paper.get("classification") not in classifications:
            continue
        title = paper.get("title") or ""
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        items.append({
            "title": title,
            "year": paper.get("year") or "",
            "source": paper.get("source") or "",
            "classification": paper.get("classification"),
            "score": paper.get("evidence_score", paper.get("score", 0.0)),
            "reason": paper.get("decision_reason") or paper.get("reason", ""),
            "matched_facets": paper.get("matched_facets", []),
            "missing_facets": paper.get("missing_facets", []),
        })
        if len(items) >= limit:
            break
    return items


def _evidence_titles(
    validation_papers: list[dict],
    classifications: set[str],
    limit: int = 3,
) -> list[str]:
    return [item["title"] for item in _evidence_items(validation_papers, classifications, limit)]


def _external_evidence_titles(validation_papers: list[dict], status: str = "", limit: int = 3) -> list[str]:
    if status == "already_addressed":
        return _evidence_titles(validation_papers, {"directly_addresses_gap"}, limit=limit)
    useful = {
        "directly_addresses_gap",
        "partially_addresses_gap",
        "related_but_not_addressing",
    }
    return _evidence_titles(validation_papers, useful, limit=limit)


def _evidence_summary(validation_papers: list[dict], status: str = "", limit: int = 3) -> list[dict]:
    if status == "already_addressed":
        useful = {"directly_addresses_gap"}
    else:
        useful = {
            "directly_addresses_gap",
            "partially_addresses_gap",
            "related_but_not_addressing",
        }
    return _evidence_items(validation_papers, useful, limit=limit)


def _has_weak_closure_evidence(status: str, validation_papers: list[dict]) -> bool:
    if status != "already_addressed":
        return False
    direct = _evidence_items(validation_papers, {"directly_addresses_gap"}, limit=5)
    if len(direct) < 2:
        return True
    return any(float(item.get("score") or 0.0) < 0.82 for item in direct)


def _classified_titles(validation_papers: list[dict], classification: str, limit: int = 5) -> list[str]:
    return _evidence_titles(validation_papers, {classification}, limit=limit)


def _classified_summary(validation_papers: list[dict], classifications: set[str], limit: int = 5) -> list[dict]:
    return _evidence_items(validation_papers, classifications, limit=limit)


def _compact_validation_result(entry: dict, result: dict) -> dict:
    refined_gap = result.get("refined_gap") or ""
    validation_papers = result.get("validation_papers") or []
    status = result.get("status") or ""
    return {
        "gap_id": entry["gap_id"],
        "gap_type": entry["gap_type"],
        "original_gap": entry["gap"],
        "status": status,
        "confidence": result.get("confidence"),
        "use_for_experiments": gap_use_for_experiments(status, refined_gap),
        "decision_reason": result.get("decision_reason", ""),
        "refined_gap": refined_gap or None,
        "artifact_path": result.get("artifact_path", ""),
        "external_evidence_titles": _external_evidence_titles(validation_papers, status=status, limit=3),
        "direct_evidence_titles": _classified_titles(validation_papers, "directly_addresses_gap", limit=5),
        "partial_evidence_titles": _classified_titles(validation_papers, "partially_addresses_gap", limit=5),
        "related_evidence_titles": _classified_titles(validation_papers, "related_but_not_addressing", limit=5),
        "evidence_summary": _evidence_summary(validation_papers, status=status, limit=3),
        "top_evidence": _classified_summary(
            validation_papers,
            {"directly_addresses_gap", "partially_addresses_gap", "related_but_not_addressing"},
            limit=5,
        ),
        "evidence_quality_warning": (
            "Some validation evidence may be adjacent rather than directly addressing the gap."
            if _has_weak_closure_evidence(status, validation_papers)
            else ""
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
        if progress_callback:
            progress_callback("running", entry, None, None)
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
    _BATCH_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_slug = _safe_slug(summary["project"], max_len=60)
    path = _BATCH_VALIDATION_DIR / f"batch_gap_validation_{timestamp}_{project_slug}.json"
    summary["batch_artifact_path"] = str(path)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
                "external_evidence_titles": (
                    (item.get("direct_evidence_titles") or [])[:3]
                    or (item.get("partial_evidence_titles") or [])[:3]
                    or (item.get("external_evidence_titles") or [])[:3]
                ),
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
    project_context = project or ""
    gap_facets = extract_gap_facets(gap, project_context=project_context)
    search_queries = generate_gap_validation_queries(gap, project_context=project_context)
    per_query_limit = max(1, min(max_results, 10))
    search_results = _search_gap_queries(search_queries, per_query_limit)
    validation_papers = [
        _normalize_validation_paper(paper, gap, project_ids, project_context=project_context)
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
        "gap_facets": gap_facets,
        "results_found": len(validation_papers),
        "relevant_results": relevant_results,
        "status": status,
        "confidence": confidence,
        "decision_reason": decision_reason,
        "refined_gap": refined_gap,
        "validation_papers": validation_papers,
        "direct_evidence_titles": _classified_titles(validation_papers, "directly_addresses_gap", limit=5),
        "partial_evidence_titles": _classified_titles(validation_papers, "partially_addresses_gap", limit=5),
        "related_evidence_titles": _classified_titles(validation_papers, "related_but_not_addressing", limit=5),
        "evidence_summary": _evidence_summary(validation_papers, status=status, limit=5),
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
