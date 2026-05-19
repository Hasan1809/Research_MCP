"""MCP tool for research gap detection across multiple papers."""
import json
from datetime import datetime

from config import DATA_DIR
from services.analysis.gap_detector import detect_gaps
from services.paper_repository import load_profile_or_insights
from services.project_manager import get_project_papers
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_ANALYSIS_DIR = DATA_DIR / "analysis"


def _load_paper_data(paper_id: str, source: str) -> dict:
    logger.info("Loading profile or insights for paper_id=%r", paper_id)
    return load_profile_or_insights(source, paper_id)


def detect_gaps_tool(papers: list[dict] = None, project: str = None) -> dict:
    """
    Identify research gaps across multiple profiled papers.

    Requires paper profiles to have been built for each paper first.
    Pass papers as a list of dicts: [{"paper_id": "2602.07652", "source": "arxiv"}, ...]
    Requires at least 2 papers.

    Example:
      detect_gaps_tool(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "2603.17419", "source": "arxiv"}
      ])

    Returns research_gaps, methodological_gaps, contradictions,
    connections, and field_summary.
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

    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(ref.get("paper_id", "?").replace("/", "-") for ref in papers[:3])
    save_path = _ANALYSIS_DIR / f"gap_analysis_{timestamp}_{paper_ids}.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump({"papers": papers, "analysis": result}, f, indent=2)
    logger.info("Gap analysis saved to %s", save_path)

    log_invocation("detect_gaps_tool", arguments, output=result)
    return result
