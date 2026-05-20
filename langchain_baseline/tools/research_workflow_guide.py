from langchain_core.tools import tool

from langchain_baseline.services import research_workflow_guide_impl


@tool
def get_research_workflow_guide(
    topic: str,
    project: str | None = None,
    num_papers: int = 8,
) -> dict:
    """Call first for normal requests like 'find gaps and suggest experiments for <topic>'."""
    return research_workflow_guide_impl(topic=topic, project=project, num_papers=num_papers)
