"""MCP tool for suggesting experiments based on gap analysis."""
import json
import os
from datetime import datetime
from services.analysis.experiment_suggester import suggest_experiments
from services.analysis.gap_detector import detect_gaps
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "profiles")
_ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "analysis")


def suggest_experiments_tool(
    papers: list[dict] = None,
    project: str = None,
    gap_analysis: dict = None,
) -> dict:
    """
    Suggest concrete experiments based on research gaps across papers.

    Runs gap detection internally, then generates 3-5 experiment
    proposals with hypotheses, methods, baselines, and feasibility.

    Call with project="name" to use all papers in a project.
    Each paper must have been profiled first.
    Pass gap_analysis to skip the internal gap detection step (saves ~15s).
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

    # Load profiles for all papers
    profiles = []
    for ref in papers:
        paper_id = ref.get("paper_id")
        source = ref.get("source")
        profile_path = os.path.join(_PROFILES_DIR, source, f"{paper_id}.json")
        if not os.path.exists(profile_path):
            error = (
                f"Profile not found for paper_id={paper_id!r} source={source!r}. "
                "Run build_paper_profile_tool first."
            )
            log_invocation("suggest_experiments_tool", arguments, error=error)
            raise FileNotFoundError(error)
        with open(profile_path, encoding="utf-8") as f:
            profiles.append(json.load(f))

    # Run gap detection (or use provided analysis)
    if gap_analysis is None:
        logger.info("Running gap detection for %d papers...", len(profiles))
        try:
            gap_analysis, _ = detect_gaps(profiles)
        except Exception as e:
            log_invocation("suggest_experiments_tool", arguments, error=str(e))
            raise
    else:
        logger.info("Using provided gap analysis (skipping detection)")

    # Generate experiment suggestions
    logger.info("Generating experiment suggestions...")
    try:
        result, raw = suggest_experiments(gap_analysis, profiles)
    except Exception as e:
        log_invocation("suggest_experiments_tool", arguments, error=str(e))
        raise

    # Persist to data/analysis/
    os.makedirs(_ANALYSIS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(
        ref.get("paper_id", "?").replace("/", "-") for ref in papers[:3]
    )
    save_path = os.path.join(_ANALYSIS_DIR, f"experiments_{timestamp}_{paper_ids}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"papers": papers, "gaps": gap_analysis, "experiments": result}, f, indent=2)
    logger.info("Experiment suggestions saved to %s", save_path)

    experiments = result.get("experiments", [])
    log_invocation("suggest_experiments_tool", arguments, output={
        "experiment_count": len(experiments),
        "gap_count": len(gap_analysis.get("research_gaps", [])),
        "save_path": save_path,
    })

    return {
        "gaps": gap_analysis,
        "experiments": experiments,
    }
