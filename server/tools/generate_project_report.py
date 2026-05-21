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
    Omit optional artifact path arguments unless you have a real file path; do
    not pass the string "null".
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
        log_invocation("generate_project_report_tool", arguments, output={
            "project": result.get("project"),
            "report_path": result.get("report_path"),
            "paper_count": result.get("paper_count"),
            "gap_count": result.get("gap_count"),
            "included_validated_gap_count": result.get("included_validated_gap_count"),
            "excluded_validated_gap_count": result.get("excluded_validated_gap_count"),
            "experiment_count": result.get("experiment_count"),
            "bibliography_path": result.get("bibliography_path"),
            "report_markdown_chars": len(result.get("report_markdown") or ""),
            "error": result.get("error"),
        })
        return result
    except Exception as e:
        log_invocation("generate_project_report_tool", arguments, error=str(e))
        raise
