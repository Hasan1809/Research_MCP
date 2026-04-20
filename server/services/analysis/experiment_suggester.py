"""LLM-powered experiment suggestion based on gap analysis."""
import json
import os
import time
import httpx
from services.extraction.llm_extractor import _strip_code_fences
from utils.logger import get_logger
from utils.usage_tracker import log_usage

logger = get_logger(__name__)

_EXPERIMENT_SYSTEM_PROMPT = """\
You are a research advisor. Given an analysis of research gaps across \
multiple papers and the paper profiles themselves, suggest concrete \
experiments that could address the identified gaps.

Rules:
- Each experiment must directly address a specific identified gap.
- Experiments must be feasible for a research team with standard compute \
resources (not "train a 100B model from scratch").
- Propose specific methods, baselines, and evaluation criteria.
- Reference actual paper_ids when building on existing work.
- Rate feasibility honestly: "high" means doable in weeks, "medium" in \
months, "low" means requires significant resources or novel techniques.
- Suggest 3-5 experiments, prioritized by impact and feasibility.
- Do NOT suggest vague directions like "explore more". Each experiment \
must have a testable hypothesis.

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
Here is a gap analysis across {n} research papers, followed by the \
paper profiles themselves.

--- GAP ANALYSIS ---
{gap_analysis_text}

--- PAPER PROFILES ---
{paper_profiles_text}

Suggest concrete experiments to address the identified gaps. Return \
the JSON object now.\
"""


def suggest_experiments(gap_analysis: dict, paper_profiles: list[dict]) -> tuple[dict, str]:
    """
    Generate experiment suggestions based on identified research gaps.

    Args:
        gap_analysis: Output of detect_gaps() — research_gaps,
                      methodological_gaps, contradictions, connections
        paper_profiles: The profile dicts used for gap detection

    Returns:
        (result_dict, raw_completion)
    """
    api_token = os.environ["IONOS_API_TOKEN"]
    base_url = os.environ["IONOS_BASE_URL"].rstrip("/")
    model = os.environ["IONOS_MODEL"]

    gap_text = json.dumps(
        {
            "research_gaps":       gap_analysis.get("research_gaps", []),
            "methodological_gaps": gap_analysis.get("methodological_gaps", []),
            "contradictions":      gap_analysis.get("contradictions", []),
            "connections":         gap_analysis.get("connections", []),
        },
        indent=2,
    )

    profile_parts = []
    for p in paper_profiles:
        profile_parts.append(
            f"--- Paper: {p.get('paper_id', '?')} ({p.get('source', '?')}) ---\n"
            f"Type: {p.get('paper_type', '')}\n"
            f"Problem: {p.get('research_problem', '')}\n"
            f"Contribution: {p.get('main_contribution', '')}\n"
            f"Methods: {p.get('methods_or_approach', '')}\n"
            f"Findings: {p.get('key_findings', '')}\n"
            f"Limitations: {json.dumps(p.get('limitations', []))}"
        )
    profiles_text = "\n\n".join(profile_parts)

    user_message = _EXPERIMENT_USER_TEMPLATE.format(
        n=len(paper_profiles),
        gap_analysis_text=gap_text,
        paper_profiles_text=profiles_text,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXPERIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "response_format": {"type": "json_object"},
    }

    logger.info(
        "Suggesting experiments: model=%s papers=%d gap_chars=%d",
        model, len(paper_profiles), len(gap_text),
    )

    _start = time.time()
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if response.is_error:
        logger.error(
            "IONOS API error (experiments): status=%d body=%s",
            response.status_code, response.text,
        )
    response.raise_for_status()
    _latency = time.time() - _start
    _resp = response.json()
    _usage = _resp.get("usage", {})
    log_usage(
        tool_name="suggest_experiments",
        model=model,
        input_tokens=_usage.get("prompt_tokens", 0),
        output_tokens=_usage.get("completion_tokens", 0),
        total_tokens=_usage.get("total_tokens", 0),
        latency_seconds=_latency,
        input_chars=len(user_message),
    )

    raw = _resp["choices"][0]["message"]["content"].strip()
    logger.info("Experiment suggestion LLM response: %d chars", len(raw))
    logger.debug("Raw experiment completion: %s", raw)

    try:
        parsed = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse experiment suggestions as JSON: %s", e)
        raise

    logger.info(
        "Experiment suggestions complete: count=%d",
        len(parsed.get("experiments", [])),
    )
    return parsed, raw
