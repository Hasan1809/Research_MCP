import json
import os
import re
import httpx
from collections import Counter
from services.extraction.llm_extractor import _strip_code_fences
from utils.logger import get_logger

logger = get_logger(__name__)

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "this", "that", "is", "are", "was", "were", "be", "been",
    "by", "from", "as", "we", "our", "their", "its", "it", "using", "based",
    "via", "into", "than", "which", "also", "has", "have", "not", "no",
    "can", "paper", "study", "show", "shows", "propose", "proposed",
}

_ANALYSIS_SYSTEM_PROMPT = """\
You are a research analyst. Given titles and abstracts of multiple \
academic papers on a related topic, identify the major themes, trends, \
and methodological patterns.

Rules:
- Themes must be specific to this collection of papers, not generic.
- Each theme must reference which papers it applies to (by title).
- Trends should note temporal patterns if year data is available.
- Ground every claim in the actual titles and abstracts provided.

Return ONLY a valid JSON object with this structure:
{
  "themes": [
    {
      "theme": "named theme",
      "description": "1-2 sentence explanation",
      "papers": ["paper titles that relate"]
    }
  ],
  "trends": [
    {
      "trend": "what is changing",
      "direction": "increasing or emerging or declining",
      "evidence": "based on what"
    }
  ],
  "methodology_landscape": "2-3 sentences on how the field approaches problems",
  "paper_count": 0,
  "year_range": "e.g. 2022-2025"
}
Do not include any text outside the JSON object.\
"""

_ANALYSIS_USER_TEMPLATE = """\
Here are {n} academic papers. Analyze the themes, trends, and \
methodological patterns across them.

{papers_text}

Return the JSON analysis now.\
"""


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 3]


def _analyze_papers_fallback(papers: list[dict]) -> dict:
    """Word-frequency fallback used when the LLM call fails."""
    all_words: list[str] = []
    years: list[int] = []
    sources: list[str] = []

    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        all_words.extend(_tokenize(text))
        if paper.get("year"):
            try:
                years.append(int(paper["year"]))
            except ValueError:
                pass
        if paper.get("source"):
            sources.append(paper["source"])

    word_counts = Counter(all_words)
    top_words = [word for word, _ in word_counts.most_common(30)]

    themes = top_words[:5]
    common_topics = top_words[5:15]

    limitations = []
    if len(papers) < 3:
        limitations.append("Too few papers to draw reliable conclusions")
    if years and (max(years) - min(years)) < 2:
        limitations.append("Papers span a narrow time range; trends may not be visible")
    if len(set(sources)) == 1:
        limitations.append(f"All papers from a single source ({sources[0]}); results may be biased")

    possible_gaps = []
    rare_words = [word for word, count in word_counts.items() if count == 1 and len(word) > 5]
    if rare_words:
        possible_gaps.append(f"Underexplored concepts: {', '.join(rare_words[:5])}")
    if years and max(years) < 2023:
        possible_gaps.append("No recent papers found; the field may have advanced since")

    return {
        "themes": themes,
        "common_topics": common_topics,
        "limitations": limitations,
        "possible_gaps": possible_gaps,
    }


def analyze_papers(papers: list[dict]) -> dict:
    """LLM-powered paper analysis with word-frequency fallback."""
    try:
        api_token = os.environ["IONOS_API_TOKEN"]
        base_url = os.environ["IONOS_BASE_URL"].rstrip("/")
        model = os.environ["IONOS_MODEL"]
    except KeyError as e:
        logger.warning("IONOS env var missing (%s) — using fallback analysis", e)
        return _analyze_papers_fallback(papers)

    papers_text = "\n\n".join(
        f"Title: {p.get('title', '')}\n"
        f"Year: {p.get('year', 'unknown')}\n"
        f"Abstract: {p.get('abstract', '')}"
        for p in papers
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _ANALYSIS_USER_TEMPLATE.format(
                    n=len(papers), papers_text=papers_text
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    logger.info("Analyzing %d papers with LLM: model=%s", len(papers), model)

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.is_error:
            logger.error(
                "IONOS API error (analyze_papers): status=%d body=%s",
                response.status_code, response.text,
            )
        response.raise_for_status()
    except Exception as e:
        logger.warning("LLM analysis failed (%s) — using fallback", e)
        return _analyze_papers_fallback(papers)

    raw = response.json()["choices"][0]["message"]["content"].strip()
    logger.info("LLM analysis response: %d chars", len(raw))

    try:
        parsed = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM analysis response (%s) — using fallback", e)
        return _analyze_papers_fallback(papers)

    logger.info(
        "LLM analysis complete: themes=%d trends=%d",
        len(parsed.get("themes", [])),
        len(parsed.get("trends", [])),
    )
    return parsed


def _most_common_items(lists: list[list[str]], threshold: int) -> list[str]:
    """Return items that appear in at least `threshold` of the provided lists."""
    counts: Counter = Counter()
    for items in lists:
        for item in set(item.lower().strip() for item in items):
            counts[item] += 1
    return [item for item, count in counts.most_common() if count >= threshold]


def synthesize_insights(insights: list[dict]) -> dict:
    n = len(insights)
    min_overlap = max(2, n // 2)  # item must appear in at least half the papers (min 2)

    methods_lists   = [i.get("methods", []) for i in insights]
    datasets_lists  = [i.get("datasets", []) for i in insights]
    limits_lists    = [i.get("limitations", []) for i in insights]
    results_lists   = [i.get("results", []) for i in insights]

    common_methods      = _most_common_items(methods_lists, min_overlap)
    common_datasets     = _most_common_items(datasets_lists, min_overlap)
    recurring_limitations = _most_common_items(limits_lists, min_overlap)
    common_findings     = _most_common_items(results_lists, min_overlap)

    # Notable differences: items unique to a single paper
    all_methods = [item.lower().strip() for lst in methods_lists for item in lst]
    method_counts = Counter(all_methods)
    notable_differences = [item for item, count in method_counts.items() if count == 1][:5]

    logger.info(
        "Synthesis over %d papers: common_methods=%d common_datasets=%d "
        "recurring_limitations=%d common_findings=%d notable_differences=%d",
        n, len(common_methods), len(common_datasets),
        len(recurring_limitations), len(common_findings), len(notable_differences),
    )

    return {
        "common_methods": common_methods,
        "common_datasets": common_datasets,
        "recurring_limitations": recurring_limitations,
        "common_findings": common_findings,
        "notable_differences": notable_differences,
    }
