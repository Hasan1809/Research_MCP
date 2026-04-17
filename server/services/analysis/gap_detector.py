"""
LLM-powered research gap detection across multiple papers.

Takes paper profiles (from build_paper_profile_tool) or insights
(from extract_paper_insights_tool) for multiple papers and sends them
to the LLM to identify:
- Research gaps: topics not adequately covered
- Methodological gaps: approaches not tried
- Contradictions: conflicting findings between papers
- Connections: unexpected links between papers
"""
import json
import os
import httpx
from services.extraction.llm_extractor import _strip_code_fences
from utils.logger import get_logger

logger = get_logger(__name__)

_GAP_DETECTION_SYSTEM_PROMPT = """\
You are a research analyst examining a collection of academic papers on a \
related topic. Your job is to identify what is MISSING from this body of work, \
not to summarize what is present.

You will receive structured profiles of multiple papers. Analyze them to find:

1. RESEARCH GAPS: Important questions or problems that none of the papers \
adequately address. Look for:
   - Topics mentioned in limitations/future_work that no other paper tackles
   - Assumptions shared by all papers that remain untested
   - Populations, domains, or settings not covered
   - Scale or scope limitations common to all papers

2. METHODOLOGICAL GAPS: Approaches or techniques that could be applied but \
haven't been. Look for:
   - Methods used in paper A's domain that could benefit paper B's problem
   - Evaluation approaches missing across all papers
   - Baselines or comparisons nobody has made

3. CONTRADICTIONS: Conflicting findings or claims between papers. Look for:
   - Papers reporting different results on similar tasks
   - Conflicting conclusions about the same technique
   - Incompatible assumptions or framings

4. CONNECTIONS: Non-obvious links between papers that suggest new research \
directions. Look for:
   - Complementary techniques that could be combined
   - Shared limitations that might have a common solution
   - Results in one paper that could explain findings in another

Rules:
- Every gap, contradiction, and connection MUST reference specific paper_ids.
- Be specific and concrete. "More research is needed" is NOT a valid gap.
- Only identify gaps that are genuinely supported by the evidence in the profiles.
- If you cannot identify items for a category, return an empty list.
- Do NOT simply restate items from the future_work or limitations fields of the paper \
profiles. Those are the authors' own suggestions. Your job is to identify gaps they \
MISSED — things that none of the papers recognized as a gap but that become visible \
when comparing across papers.

Return ONLY a valid JSON object with exactly these keys:
{
  "research_gaps": [
    {
      "gap": "string — what is missing",
      "evidence": "string — why this is a gap based on the papers",
      "relevant_papers": ["paper_id", ...]
    }
  ],
  "methodological_gaps": [
    {
      "gap": "string",
      "current_approaches": ["string", ...],
      "missing_approach": "string — what hasn't been tried",
      "relevant_papers": ["paper_id", ...]
    }
  ],
  "contradictions": [
    {
      "finding_a": "string",
      "finding_b": "string",
      "paper_a": "paper_id",
      "paper_b": "paper_id",
      "nature": "string — why they conflict"
    }
  ],
  "connections": [
    {
      "insight": "string — the connection found",
      "papers": ["paper_id", ...],
      "potential": "string — what this connection enables"
    }
  ],
  "field_summary": "string — 2-3 sentence summary of the state of this research area"
}

No text outside the JSON object.\
"""

_GAP_DETECTION_USER_TEMPLATE = """\
Here are profiles of {n} related research papers. Analyze them to identify \
research gaps, methodological gaps, contradictions, and connections.

{paper_profiles_text}

Return the JSON analysis now.\
"""


def _format_profile(profile: dict) -> str:
    def _fmt_list(val) -> str:
        if isinstance(val, list):
            return "; ".join(str(v) for v in val) if val else "none"
        return str(val) if val else "none"

    return (
        f"--- Paper: {profile.get('paper_id', '?')} ({profile.get('source', '?')}) ---\n"
        f"Type: {profile.get('paper_type', '')}\n"
        f"Problem: {profile.get('research_problem', '')}\n"
        f"Contribution: {profile.get('main_contribution', '')}\n"
        f"Methods: {profile.get('methods_or_approach', '') or _fmt_list(profile.get('methods', []))}\n"
        f"Findings: {profile.get('key_findings', '') or _fmt_list(profile.get('results', []))}\n"
        f"Core insight: {profile.get('core_insight', '')}\n"
        f"Datasets: {_fmt_list(profile.get('datasets', []))}\n"
        f"Limitations: {_fmt_list(profile.get('limitations', []))}\n"
        f"Future work: {_fmt_list(profile.get('future_work', []))}"
    )


def detect_gaps(paper_profiles: list[dict]) -> tuple[dict, str]:
    """
    Analyze multiple paper profiles to identify research gaps.

    Returns (result_dict, raw_completion) where result_dict has keys:
    research_gaps, methodological_gaps, contradictions, connections, field_summary
    """
    api_token = os.environ["IONOS_API_TOKEN"]
    base_url = os.environ["IONOS_BASE_URL"].rstrip("/")
    model = os.environ["IONOS_MODEL"]

    paper_profiles_text = "\n\n".join(_format_profile(p) for p in paper_profiles)
    user_message = _GAP_DETECTION_USER_TEMPLATE.format(
        n=len(paper_profiles),
        paper_profiles_text=paper_profiles_text,
    )
    total_chars = len(user_message)
    logger.info(
        "Gap detection: %d papers, total_chars=%d", len(paper_profiles), total_chars
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _GAP_DETECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "response_format": {"type": "json_object"},
    }

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if response.is_error:
        logger.error("IONOS API error (gap_detection): status=%d body=%s",
                     response.status_code, response.text)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()
    logger.info("Gap detection LLM response: %d chars", len(raw))
    logger.debug("Raw gap detection completion: %s", raw)

    try:
        parsed = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse gap detection response as JSON: %s", e)
        raise

    result = {
        "research_gaps":       parsed.get("research_gaps") or [],
        "methodological_gaps": parsed.get("methodological_gaps") or [],
        "contradictions":      parsed.get("contradictions") or [],
        "connections":         parsed.get("connections") or [],
        "field_summary":       parsed.get("field_summary") or "",
    }

    logger.info(
        "Gap detection complete: research_gaps=%d methodological_gaps=%d "
        "contradictions=%d connections=%d",
        len(result["research_gaps"]),
        len(result["methodological_gaps"]),
        len(result["contradictions"]),
        len(result["connections"]),
    )
    return result, raw
