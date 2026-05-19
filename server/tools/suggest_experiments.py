"""MCP tool for suggesting experiments based on gap analysis."""
from services.analysis.experiment_suggester import suggest_experiments_for_papers
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def suggest_experiments_tool(
    papers: list[dict] = None,
    project: str = None,
    gap_analysis: dict = None,
    compact: bool = True,
) -> dict:
    """
    Suggest concrete research experiments from an existing project or explicit paper list.

    Prefer passing project when papers are already grouped in a saved project; the tool
    will load project papers internally. Do not pass both project and papers unless
    necessary. If both are provided, project wins and the explicit paper list is ignored.
    Use papers=[{"paper_id": "...", "source": "..."}] only for one-off analysis without
    a saved project.

    In compact mode, the tool reuses cached gap analysis when available and generates
    experiments from gaps plus minimal paper metadata instead of full profiles.

    Requires saved paper profiles when no matching gap analysis cache exists and gap
    detection must run. Requires at least 2 active papers.
    Always call this after detect_gaps_tool, not instead of it.

    Preferred project example:
      suggest_experiments_tool(project="llm-agent-security-batch-project-test", compact=True)

    Explicit paper-list example:
      suggest_experiments_tool(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "2603.17419", "source": "arxiv"}
      ], compact=True)

    Returns 3-5 experiment proposals with hypotheses, methods,
    baselines, datasets, feasibility ratings, and compact run metadata.
    """
    if project:
        from services.project_manager import get_project_papers
        if papers:
            logger.warning(
                "Both project and papers provided; using project manifest and ignoring papers."
            )
        papers = get_project_papers(project)
        logger.info(
            "Tool invoked: suggest_experiments project=%r active_paper_count=%d compact=%s",
            project,
            len(papers),
            compact,
        )
    elif papers:
        logger.info(
            "Tool invoked: suggest_experiments active_paper_count=%d compact=%s",
            len(papers),
            compact,
        )
    else:
        raise ValueError("suggest_experiments_tool requires either 'papers' or 'project'.")

    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for experiment suggestions.")

    if project:
        arguments = {
            "project": project,
            "active_paper_count": len(papers),
            "compact": compact,
        }
    else:
        arguments = {
            "papers": papers,
            "active_paper_count": len(papers),
            "compact": compact,
        }

    try:
        result = suggest_experiments_for_papers(
            papers,
            gap_analysis=gap_analysis,
            compact=compact,
            project=project,
        )
    except Exception as e:
        log_invocation("suggest_experiments_tool", arguments, error=str(e))
        raise

    experiments = result.get("experiments", [])
    output = {
        "experiment_count": len(experiments),
        "gap_count": result.get("gap_count"),
        "gap_source": result.get("gap_source"),
        "project": project,
        "active_paper_count": len(papers),
        "compact": compact,
        "gap_analysis_path": result.get("gap_analysis_path"),
        "save_path": result.get("save_path"),
        "error": result.get("error"),
        "validation_used": result.get("validation_used"),
        "batch_validation_path": result.get("batch_validation_path"),
        "included_gap_count": result.get("included_gap_count"),
        "excluded_gap_count": result.get("excluded_gap_count"),
        "refined_gap_count": result.get("refined_gap_count"),
    }
    logger.info(
        "suggest_experiments complete: project=%r active_paper_count=%d compact=%s gap_source=%s save_path=%s",
        project,
        len(papers),
        compact,
        result.get("gap_source"),
        result.get("save_path"),
    )
    log_invocation("suggest_experiments_tool", arguments, output=output)

    return result
