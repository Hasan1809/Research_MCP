"""MCP tools for managing research projects."""
from services.project_manager import (
    create_project,
    batch_add_papers_to_project,
    clear_project,
)
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def create_project_tool(name: str, overwrite: bool = False) -> dict:
    """
    Create a new research project that groups papers by topic.

    Args:
        name: Project name (e.g. "moe-efficiency"). Will be slugified
              to lowercase with hyphens.
        overwrite: If true, replace an existing project with an empty manifest.

    Returns:
        The project manifest dict with name, created, and papers list.
    """
    logger.info("Tool invoked: create_project name=%r overwrite=%s", name, overwrite)
    arguments = {"name": name, "overwrite": overwrite}
    try:
        result = create_project(name, overwrite=overwrite)
        log_invocation("create_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("create_project_tool", arguments, error=str(e))
        raise

def batch_add_to_project_tool(name: str, papers: list[dict]) -> dict:
    """
    Add multiple papers to a project in one operation.

    Args:
        name: Project name. Will be slugified to match project manifests.
        papers: List of {"paper_id": "...", "source": "..."} references.

    Returns:
        Structured result with added, skipped, duplicate, failed, and summary counts.
    """
    logger.info("Tool invoked: batch_add_to_project name=%r count=%d", name, len(papers or []))
    arguments = {"name": name, "papers": papers}
    try:
        result = batch_add_papers_to_project(name, papers)
        log_invocation("batch_add_to_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("batch_add_to_project_tool", arguments, error=str(e))
        raise

def clear_project_tool(name: str) -> dict:
    """
    Remove all papers from an existing project manifest.

    Args:
        name: Project name. Will be slugified to match project manifests.

    Returns:
        Structured result with removed_count and updated manifest.
    """
    logger.info("Tool invoked: clear_project name=%r", name)
    arguments = {"name": name}
    try:
        result = clear_project(name)
        log_invocation("clear_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("clear_project_tool", arguments, error=str(e))
        raise
