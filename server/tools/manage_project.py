"""MCP tools for managing research projects."""
from services.project_manager import (
    create_project,
    add_paper_to_project,
    list_projects,
)
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def create_project_tool(name: str) -> dict:
    """
    Create a new research project that groups papers by topic.

    Args:
        name: Project name (e.g. "moe-efficiency"). Will be slugified
              to lowercase with hyphens.

    Returns:
        The project manifest dict with name, created, and papers list.
    """
    logger.info("Tool invoked: create_project name=%r", name)
    arguments = {"name": name}
    try:
        result = create_project(name)
        log_invocation("create_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("create_project_tool", arguments, error=str(e))
        raise


def add_to_project_tool(name: str, paper_id: str, source: str) -> dict:
    """
    Add a paper to a project. The paper should already be ingested.

    Args:
        name:     Project name.
        paper_id: Paper ID (e.g. "2401.04088" for arXiv).
        source:   Source, e.g. "arxiv".

    Returns:
        Updated project manifest.
    """
    logger.info("Tool invoked: add_to_project name=%r paper_id=%r source=%r", name, paper_id, source)
    arguments = {"name": name, "paper_id": paper_id, "source": source}
    try:
        result = add_paper_to_project(name, paper_id, source)
        log_invocation("add_to_project_tool", arguments, output={
            "name": result["name"],
            "paper_count": len(result["papers"]),
        })
        return result
    except Exception as e:
        log_invocation("add_to_project_tool", arguments, error=str(e))
        raise


def list_projects_tool() -> list[dict]:
    """
    List all research projects with their paper counts.

    Returns:
        List of dicts with name, created, paper_count, and papers.
    """
    logger.info("Tool invoked: list_projects")
    try:
        result = list_projects()
        log_invocation("list_projects_tool", {}, output=result)
        return result
    except Exception as e:
        log_invocation("list_projects_tool", {}, error=str(e))
        raise
