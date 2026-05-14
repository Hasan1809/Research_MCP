from services.citations import generate_bibliography
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def generate_bibliography_tool(
    project_name: str = "",
    papers: list[dict] | None = None,
    paper_ids: list[str] | None = None,
    source: str = "arxiv",
    format: str = "bibtex",
    save: bool = True,
) -> dict:
    """
    Generate a bibliography from a project or explicit paper references.

    Args:
        project_name: Optional saved project name. If provided, its papers are used.
        papers: Optional list of {"paper_id": "...", "source": "..."} refs.
        paper_ids: Optional list of paper IDs, using source for all entries.
        source: Source to use with paper_ids. Defaults to "arxiv".
        format: One of "bibtex", "markdown", "ieee", "plaintext", or "plain".
        save: Whether to persist an export under data/artifacts/bibliographies.

    Returns:
        Dict with bibliography string, included metadata records, skipped papers,
        and artifact_path if saved.
    """
    arguments = {
        "project_name": project_name,
        "papers": papers,
        "paper_ids": paper_ids,
        "source": source,
        "format": format,
        "save": save,
    }
    logger.info(
        "Tool invoked: generate_bibliography project=%r format=%r",
        project_name, format,
    )
    try:
        result = generate_bibliography(
            project_name=project_name or None,
            papers=papers,
            paper_ids=paper_ids,
            source=source,
            format=format,
            save=save,
        )
        log_invocation("generate_bibliography_tool", arguments, output={
            "format": result["format"],
            "included_count": len(result["included"]),
            "skipped_count": len(result["skipped"]),
            "artifact_path": result["artifact_path"],
        })
        return result
    except Exception as e:
        log_invocation("generate_bibliography_tool", arguments, error=str(e))
        raise
