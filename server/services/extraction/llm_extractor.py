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


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


_SYSTEM_PROMPT = """\
You are a scientific information extractor. Extract information from the provided paper text.

Rules:
- Only extract information grounded in the provided text.
- Concise paraphrases are acceptable; do not fabricate details not present in the text.
- If a field is not present in the text, return an empty list. Do NOT infer, speculate, or generate plausible-sounding content. An empty list is always correct when the paper does not explicitly discuss that topic. Explain in debug_notes which fields are empty and why.

Return ONLY a valid JSON object with exactly these keys:
{
  "methods": [],
  "results": [],
  "datasets": [],
  "limitations": [],
  "future_work": [],
  "debug_notes": ""
}

Use debug_notes to briefly explain which fields are empty and why (e.g. "results not mentioned in provided text").
Do not include any text, explanation, or formatting outside the JSON object.\
"""

_USER_PROMPT_TEMPLATE = """\
Extract structured information from the following paper text.

--- PAPER TEXT START ---
{text}
--- PAPER TEXT END ---

Return the JSON object now.\
"""

_FIELD_SYSTEM_PROMPT = """\
You are a scientific information extractor. Extract only the "{field}" information from the provided paper text.

Rules:
- Only extract information grounded in the provided text.
- Concise paraphrases are acceptable; do not fabricate details not present in the text.
- Return an empty list if the field is not present.

Return ONLY a valid JSON object with a single key:
{{"{field}": []}}

Do not include any text, explanation, or formatting outside the JSON object.\
"""

_FIELD_SYSTEM_PROMPTS: dict[str, str] = {
    "methods": """\
You are a scientific information extractor. Extract only the "methods" information from the provided paper text.

The "methods" field should include ANY of the following if present in the text:
- experimental methods, algorithms, or techniques used
- methodological frameworks or structured approaches
- checklists or reporting guidelines
- evaluation procedures, criteria, or rubrics
- structured questions, protocols, or recommended practices

Important:
- For framework, recommendation, or review papers, treat the proposed framework, checklist, evaluation structure, or reporting procedure as methods.
- Do NOT return an empty list just because the paper is not a traditional experiment paper.
- Only extract what is explicitly stated or clearly described in the text.
- Concise paraphrases are acceptable; do not fabricate details.

Return ONLY a valid JSON object with a single key:
{"methods": []}

Do not include any text, explanation, or formatting outside the JSON object.\
""",
    "results": """\
You are a scientific information extractor. Extract only the "results" information from the provided paper text.

The "results" field should include ONLY:
- main findings or discoveries reported in the paper
- explicit conclusions stated by the authors
- reported outcomes or measured observations
- major takeaways clearly presented as findings or conclusions

The "results" field must NOT include:
- checklist items or procedural steps
- reporting recommendations or guidelines
- framework components or method descriptions (those belong in "methods")
- future work suggestions (those belong in "future_work")
- limitations (those belong in "limitations")

Important:
- For framework, recommendation, or review papers: only include explicit conclusions or reported findings - not the framework content itself.
- Keep grounding strict: only extract what is clearly stated as a finding or conclusion.
- Concise paraphrases are acceptable; do not fabricate details.
- Return an empty list if no clear results or conclusions are present.

Return ONLY a valid JSON object with a single key:
{"results": []}

Do not include any text, explanation, or formatting outside the JSON object.\
""",
}

_FIELD_USER_PROMPT_TEMPLATE = """\
Extract the "{field}" from the following paper text only.

--- PAPER TEXT START ---
{text}
--- PAPER TEXT END ---

Return the JSON object now.\
"""

_PROFILE_SYSTEM_PROMPT = """\
You are an expert research analyst. You will be given excerpts from a scientific paper.
Your job is to capture the ESSENCE and IDENTITY of this specific paper - not produce a generic academic summary.

A good profile must let a reader answer: What is this paper really about? What are the authors arguing? What makes it distinctive?

Think like a knowledgeable researcher who read the paper and wants to convey its meaning and significance to a smart colleague.

Return ONLY a valid JSON object with exactly these keys:

{
  "paper_type": "<one of: empirical study | framework paper | recommendation paper | survey | benchmark paper | position paper | perspective paper | other>",

  "research_problem": "<What specific problem is this paper addressing? Be concrete. Name the gap, failure, or challenge. Not generic - describe THIS paper's problem specifically. 2-3 sentences.>",

  "main_contribution": "<What does this paper introduce, propose, or demonstrate that did not exist before? Be specific about what is new. Avoid vague phrases like 'a new approach'. Name the actual thing. 2-3 sentences.>",

  "methods_or_approach": "<How does the paper solve the problem? What is the structure of the approach? If it is a framework or checklist, describe what it covers. If empirical, describe the evaluation design. Be specific. 2-4 sentences.>",

  "key_findings": "<What are the main conclusions or outcomes? What should a reader take away? For non-empirical papers, what is the main argument or recommendation the authors are making? 2-4 sentences.>",

  "paper_intent": "<Why does this paper exist? What is it trying to change, fix, standardize, question, or demonstrate? What gap or frustration is motivating the authors? 1-2 sentences.>",

  "core_insight": "<The single most important idea in this paper. This must be specific to THIS paper. It should be what a smart researcher would remember after reading it. Not generic. 1-2 sentences.>",

  "paper_stance": "<What position are the authors taking? What are they arguing or emphasizing? Examples: 'current reporting practices are insufficient and actively harm reproducibility', 'external data dependencies are a structural risk that is being underappreciated'. Be direct. 1-2 sentences.>",

  "distinctive_elements": [
    "<3-6 short items that make this paper recognizable and specific. Not vague. Prefer concrete identifiers: named frameworks, specific evaluation criteria, named problems, specific communities targeted, named concepts introduced.>"
  ],

  "datasets": [
    "<ONLY include if the paper explicitly uses or evaluates on a named dataset. If the paper is a framework, recommendation, or survey paper with no empirical evaluation, return an empty list. Do NOT treat frameworks, checklists, or named concepts as datasets.>"
  ],

  "limitations": ["<stated limitation from the text. Return an empty list if the paper does not explicitly discuss this.>"],

  "future_work": ["<future direction explicitly mentioned in the text. Return an empty list if the paper does not explicitly discuss this.>"],

  "plain_english_summary": "<3-5 sentences explaining this paper to a smart student with no prior context. Cover: what problem it addresses, what the authors did, and what the key takeaway is. Make it specific to THIS paper - not a generic academic paper.>"
,
  "supporting_paper_ids": ["<the current paper_id only>"],
  "citation_label": "<use [paper_id]>"
}

Rules:
- Ground everything in the provided text. Do not hallucinate.
- Concise paraphrasing is fine; do not copy large verbatim passages.
- Narrative fields (research_problem, main_contribution, etc.) must be specific and concrete, not generic.
- If a list field has no supporting text, return an empty list. Do NOT infer, speculate, or generate plausible-sounding content. An empty list is always preferred over fabricated entries. Only include items that are explicitly stated or clearly described in the text.
- Attribution rule: this profile is for exactly one paper. Use only the provided paper_id in supporting_paper_ids and citation_label. Do not cite any other paper.
- Never return empty strings for narrative fields.
- Do not include any text outside the JSON object.
"""

_PROFILE_USER_TEMPLATE = """\
Here are excerpts from research paper {paper_id}. Build a complete, specific paper profile from them.

Do NOT produce a generic academic summary. Capture what is distinctive and essential about THIS paper specifically.

--- PAPER EXCERPTS START ---
{text}
--- PAPER EXCERPTS END ---

Return the JSON profile now.\
"""


def extract_field(field: str, text: str) -> tuple[list, str]:
    system_prompt = _FIELD_SYSTEM_PROMPTS.get(field) or _FIELD_SYSTEM_PROMPT.format(field=field)
    logger.debug("System prompt for field=%r: %s", field, system_prompt)
    logger.info("Calling IONOS LLM (field=%r): model=%s text_chars=%d", field, IONOS_MODEL, len(text))

    try:
        parsed, raw = LLMClient().call(
            system=system_prompt,
            user=_FIELD_USER_PROMPT_TEMPLATE.format(field=field, text=text),
            json_mode=True,
            timeout=60,
            tool_name="extract_field",
            input_chars=len(text),
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response for field=%r - returning empty. error=%s", field, e)
        return [], ""

    logger.info("LLM response (field=%r): %d chars", field, len(raw))
    if _log_raw_llm_output():
        logger.debug("Raw completion (field=%r): %s", field, raw)
    return parsed.get(field) or [], raw


def extract_insights(text: str) -> tuple[dict, str]:
    user_message = _USER_PROMPT_TEMPLATE.format(text=text)
    logger.info("Calling IONOS LLM: model=%s text_chars=%d", IONOS_MODEL, len(text))

    parsed, raw_completion = LLMClient().call(
        system=_SYSTEM_PROMPT,
        user=user_message,
        json_mode=False,
        timeout=60,
        tool_name="extract_insights",
        input_chars=len(text),
    )

    logger.info("LLM response received: %d chars", len(raw_completion))
    if _log_raw_llm_output():
        logger.debug("Raw model completion: %s", raw_completion)

    result = {
        "methods": parsed.get("methods") or [],
        "results": parsed.get("results") or [],
        "datasets": parsed.get("datasets") or [],
        "limitations": parsed.get("limitations") or [],
        "future_work": parsed.get("future_work") or [],
        "debug_notes": parsed.get("debug_notes") or "",
    }
    return result, raw_completion


def build_profile(text: str, paper_id: str = "") -> tuple[dict, str]:
    logger.info("Building paper profile: model=%s text_chars=%d", IONOS_MODEL, len(text))

    parsed, raw = LLMClient().call(
        system=_PROFILE_SYSTEM_PROMPT,
        user=_PROFILE_USER_TEMPLATE.format(text=text, paper_id=paper_id),
        json_mode=True,
        timeout=90,
        tool_name="build_profile",
        input_chars=len(text),
        paper_id=paper_id,
    )

    logger.info("Profile LLM response: %d chars", len(raw))
    if _log_raw_llm_output():
        logger.debug("Raw profile completion: %s", raw)

    result = {
        "paper_type": parsed.get("paper_type") or "",
        "research_problem": parsed.get("research_problem") or "",
        "main_contribution": parsed.get("main_contribution") or "",
        "methods_or_approach": parsed.get("methods_or_approach") or "",
        "key_findings": parsed.get("key_findings") or "",
        "paper_intent": parsed.get("paper_intent") or "",
        "core_insight": parsed.get("core_insight") or "",
        "paper_stance": parsed.get("paper_stance") or "",
        "distinctive_elements": parsed.get("distinctive_elements") or [],
        "datasets": parsed.get("datasets") or [],
        "limitations": parsed.get("limitations") or [],
        "future_work": parsed.get("future_work") or [],
        "plain_english_summary": parsed.get("plain_english_summary") or "",
        "supporting_paper_ids": parsed.get("supporting_paper_ids") or ([paper_id] if paper_id else []),
        "citation_label": parsed.get("citation_label") or (f"[{paper_id}]" if paper_id else ""),
    }

    empty_narrative = [
        k for k in (
            "research_problem", "main_contribution", "key_findings",
            "paper_intent", "core_insight", "paper_stance",
            "plain_english_summary",
        ) if not result[k]
    ]
    if empty_narrative:
        logger.warning("Profile narrative fields returned empty: %s", empty_narrative)

    if result["datasets"]:
        logger.info("Datasets extracted: %s", result["datasets"])
    else:
        logger.info("No datasets extracted (expected for non-empirical papers)")

    return result, raw
