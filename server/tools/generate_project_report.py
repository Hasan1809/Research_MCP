from services.reports.project_report import generate_project_report
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def generate_project_report_tool(
    project: str,
    format: str = "markdown",
    gap_analysis_path: str | None = None,
    validation_batch_path: str | None = None,
    experiments_path: str | None = None,
    bibliography_path: str | None = None,
    include_bibliography: bool = True,
) -> dict:
    """
    Generates a simple deterministic Markdown report for a research project by
    combining the latest project papers, gap analysis, batch gap validation
    results, experiment suggestions, and bibliography export. This tool does
    not perform new analysis and does not call an LLM; it only summarizes
    existing saved outputs into a reusable report document.
    """
    arguments = {
        "project": project,
        "format": format,
        "gap_analysis_path": gap_analysis_path,
        "validation_batch_path": validation_batch_path,
        "experiments_path": experiments_path,
        "bibliography_path": bibliography_path,
        "include_bibliography": include_bibliography,
    }
    logger.info("Tool invoked: generate_project_report project=%r format=%r", project, format)
    try:
        result = generate_project_report(
            project=project,
            format=format,
            gap_analysis_path=gap_analysis_path,
            validation_batch_path=validation_batch_path,
            experiments_path=experiments_path,
            bibliography_path=bibliography_path,
            include_bibliography=include_bibliography,
        )
        log_invocation("generate_project_report_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("generate_project_report_tool", arguments, error=str(e))
        raise
