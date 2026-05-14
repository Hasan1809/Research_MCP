"""LLM-powered experiment suggestion based on gap analysis."""
import json
import os

from config import IONOS_MODEL
from services.extraction.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


def _log_raw_llm_output() -> bool:
    return os.environ.get("LOG_RAW_LLM_OUTPUT", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }

_EXPERIMENT_SYSTEM_PROMPT = """\
You are a research advisor. Given an analysis of research gaps across multiple papers and the paper profiles themselves, suggest concrete experiments that could address the identified gaps.

Rules:
- Each experiment must directly address a specific identified gap.
- Each experiment must cite the paper_ids or gap paper lists that motivate it.
- Only cite paper_ids that appear in the provided paper profiles. Never cite outside papers.
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
Here is a gap analysis across {n} research papers, followed by the paper profiles themselves.

--- GAP ANALYSIS ---
{gap_analysis_text}

--- PAPER PROFILES ---
{paper_profiles_text}

Suggest concrete experiments to address the identified gaps. Return the JSON object now.\
"""


def suggest_experiments(gap_analysis: dict, paper_profiles: list[dict]) -> tuple[dict, str]:
    allowed_ids = {p.get("paper_id") for p in paper_profiles if p.get("paper_id")}
    gap_text = json.dumps(
        {
            "research_gaps": gap_analysis.get("research_gaps", []),
            "methodological_gaps": gap_analysis.get("methodological_gaps", []),
            "contradictions": gap_analysis.get("contradictions", []),
            "connections": gap_analysis.get("connections", []),
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

    logger.info(
        "Suggesting experiments: model=%s papers=%d gap_chars=%d",
        IONOS_MODEL, len(paper_profiles), len(gap_text),
    )

    parsed, raw = LLMClient().call(
        system=_EXPERIMENT_SYSTEM_PROMPT,
        user=user_message,
        json_mode=True,
        timeout=90,
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
    for experiment in parsed.get("experiments", []):
        experiment["builds_on"] = [
            pid for pid in experiment.get("builds_on", []) if pid in allowed_ids
        ]
    return parsed, raw
