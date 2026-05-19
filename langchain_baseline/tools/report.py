from langchain_core.tools import tool

from langchain_baseline.services import generate_project_report_impl


@tool
def generate_project_report(
    project: str,
    format: str = "markdown",
    gap_analysis_path: str | None = None,
    validation_batch_path: str | None = None,
    experiments_path: str | None = None,
    bibliography_path: str | None = None,
    include_bibliography: bool = True,
) -> dict:
    """
    Generate a deterministic Markdown report from saved workflow artifacts.

    This does not call an LLM. It combines the project manifest, gap analysis,
    batch validation, experiment suggestions, and bibliography artifacts.
    """
    return generate_project_report_impl(
        project=project,
        format=format,
        gap_analysis_path=gap_analysis_path,
        validation_batch_path=validation_batch_path,
        experiments_path=experiments_path,
        bibliography_path=bibliography_path,
        include_bibliography=include_bibliography,
    )
