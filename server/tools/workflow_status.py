"""MCP tool for checking modular research workflow state."""

from services.workflow_status import get_workflow_status
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def get_workflow_status_tool(project: str) -> dict:
    """
    Inspect saved project state and artifacts, then recommend the next workflow step.

    Use this after creating a project, after long-running jobs, or when a Claude
    Desktop session reconnects and needs to resume without repeating work.
    """
    logger.info("Tool invoked: get_workflow_status project=%r", project)
    arguments = {"project": project}
    try:
        result = get_workflow_status(project)
        log_invocation(
            "get_workflow_status_tool",
            arguments,
            output={
                "project": result.get("project"),
                "paper_count": result.get("paper_count"),
                "profiled_count": result.get("profiled_count"),
                "next_tool": (result.get("next_step") or {}).get("tool"),
            },
        )
        return result
    except Exception as e:
        log_invocation("get_workflow_status_tool", arguments, error=str(e))
        raise
