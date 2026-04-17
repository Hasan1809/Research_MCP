"""MCP tool for research gap detection across multiple papers."""
import json
import os
from datetime import datetime
from services.analysis.gap_detector import detect_gaps
from services.project_manager import get_project_papers
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "profiles")
_INSIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "insights")
_ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "analysis")


def _load_paper_data(paper_id: str, source: str) -> dict:
    profile_path = os.path.join(_PROFILES_DIR, source, f"{paper_id}.json")
    insights_path = os.path.join(_INSIGHTS_DIR, source, f"{paper_id}.json")

    if os.path.exists(profile_path):
        logger.info("Loading profile for paper_id=%r", paper_id)
        with open(profile_path, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(insights_path):
        logger.info("Loading insights (no profile) for paper_id=%r", paper_id)
        with open(insights_path, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No profile or insights found for paper_id={paper_id!r} source={source!r}. "
        "Run build_paper_profile_tool first."
    )


def detect_gaps_tool(papers: list[dict] = None, project: str = None) -> dict:
    """
    Detect research gaps across multiple papers.

    Can be called two ways:
    - papers=[{"paper_id": ..., "source": ...}, ...] — explicit list
    - project="moe-efficiency" — use all papers from the named project

    If both are provided, project takes precedence.
    Each paper must have been profiled already
    (build_paper_profile_tool must have been run on each).

    Returns:
        Gap analysis result dict with research_gaps, methodological_gaps,
        contradictions, connections, and field_summary.
    """
    if project:
        papers = get_project_papers(project)
        logger.info("Tool invoked: detect_gaps project=%r count=%d", project, len(papers))
    elif papers:
        logger.info("Tool invoked: detect_gaps count=%d", len(papers))
    else:
        raise ValueError("detect_gaps_tool requires either 'papers' or 'project'.")

    arguments = {"papers": papers, "project": project}

    if len(papers) < 2:
        error = "detect_gaps_tool requires at least 2 papers."
        log_invocation("detect_gaps_tool", arguments, error=error)
        raise ValueError(error)

    profiles = []
    for ref in papers:
        paper_id = ref.get("paper_id")
        source = ref.get("source")
        try:
            profiles.append(_load_paper_data(paper_id, source))
        except FileNotFoundError as e:
            log_invocation("detect_gaps_tool", arguments, error=str(e))
            raise

    try:
        result, raw = detect_gaps(profiles)
    except Exception as e:
        log_invocation("detect_gaps_tool", arguments, error=str(e))
        raise

    # Save result to data/analysis/
    os.makedirs(_ANALYSIS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(ref.get("paper_id", "?").replace("/", "-") for ref in papers[:3])
    save_path = os.path.join(_ANALYSIS_DIR, f"gap_analysis_{timestamp}_{paper_ids}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"papers": papers, "analysis": result}, f, indent=2)
    logger.info("Gap analysis saved to %s", save_path)

    log_invocation("detect_gaps_tool", arguments, output=result)
    return result
