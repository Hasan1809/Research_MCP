"""MCP tool for suggesting experiments based on gap analysis."""
import json
from datetime import datetime

from config import DATA_DIR
from services.analysis.experiment_suggester import suggest_experiments
from services.analysis.gap_detector import detect_gaps
from services.paper_repository import load_profile
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_ANALYSIS_DIR = DATA_DIR / "analysis"


def _find_existing_gap_analysis(paper_ids: list[str]) -> dict | None:
    sorted_ids = sorted(paper_ids)
    for path in sorted(_ANALYSIS_DIR.glob("gap_analysis_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing_ids = sorted(p["paper_id"] for p in data.get("papers", []))
            if existing_ids == sorted_ids:
                return data["analysis"]
        except Exception:
            continue
    return None


def suggest_experiments_tool(
    papers: list[dict] = None,
    project: str = None,
    gap_analysis: dict = None,
) -> dict:
    """
    Suggest concrete experiments based on research gaps across papers.

    Requires build_paper_profile_tool to have been called for each paper first.
    Pass papers as a list of dicts: [{"paper_id": "2602.07652", "source": "arxiv"}, ...]
    Requires at least 2 papers.
    Always call this after detect_gaps_tool, not instead of it.

    Example:
      suggest_experiments_tool(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "2603.17419", "source": "arxiv"}
      ])

    Returns 3-5 experiment proposals with hypotheses, methods,
    baselines, datasets, and feasibility ratings.
    """
    if project:
        from services.project_manager import get_project_papers
        papers = get_project_papers(project)
        logger.info("Loaded %d papers from project %r", len(papers), project)
    elif papers:
        logger.info("Tool invoked: suggest_experiments count=%d", len(papers))
    else:
        raise ValueError("suggest_experiments_tool requires either 'papers' or 'project'.")

    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for experiment suggestions.")

    arguments = {"papers": papers, "project": project}

    profiles = []
    for ref in papers:
        paper_id = ref.get("paper_id")
        source = ref.get("source")
        profile = load_profile(source, paper_id)
        if profile is None:
            error = (
                f"Profile not found for paper_id={paper_id!r} source={source!r}. "
                "Run build_paper_profile_tool first."
            )
            log_invocation("suggest_experiments_tool", arguments, error=error)
            raise FileNotFoundError(error)
        profiles.append(profile)

    if gap_analysis is None:
        paper_ids = [ref.get("paper_id", "") for ref in papers]
        cached_gap_analysis = _find_existing_gap_analysis(paper_ids)
        if cached_gap_analysis is not None:
            gap_analysis = cached_gap_analysis
            logger.info("Gap analysis cache hit for paper_ids=%s", sorted(paper_ids))
        else:
            logger.info("Gap analysis cache miss for paper_ids=%s", sorted(paper_ids))
            logger.info("Running gap detection for %d papers...", len(profiles))
            try:
                gap_analysis, _ = detect_gaps(profiles)
            except Exception as e:
                log_invocation("suggest_experiments_tool", arguments, error=str(e))
                raise
    else:
        logger.info("Using provided gap analysis (skipping detection)")

    logger.info("Generating experiment suggestions...")
    try:
        result, raw = suggest_experiments(gap_analysis, profiles)
    except Exception as e:
        log_invocation("suggest_experiments_tool", arguments, error=str(e))
        raise

    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(ref.get("paper_id", "?").replace("/", "-") for ref in papers[:3])
    save_path = _ANALYSIS_DIR / f"experiments_{timestamp}_{paper_ids}.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump({"papers": papers, "gaps": gap_analysis, "experiments": result}, f, indent=2)
    logger.info("Experiment suggestions saved to %s", save_path)

    experiments = result.get("experiments", [])
    log_invocation("suggest_experiments_tool", arguments, output={
        "experiment_count": len(experiments),
        "gap_count": len(gap_analysis.get("research_gaps", [])),
        "save_path": str(save_path),
    })

    return {
        "gaps": gap_analysis,
        "experiments": experiments,
    }
