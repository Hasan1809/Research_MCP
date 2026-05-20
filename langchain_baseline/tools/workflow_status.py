from langchain_core.tools import tool

from langchain_baseline.services import get_workflow_status_impl


@tool
def get_workflow_status(project: str) -> dict:
    """Inspect project artifacts and return the next recommended workflow tool."""
    return get_workflow_status_impl(project=project)
