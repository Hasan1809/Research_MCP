import json

from langchain_core.tools import tool

from langchain_baseline.services import (
    add_to_project_impl,
    batch_add_to_project_impl,
    create_project_impl,
    list_projects_impl,
)


@tool
def create_project(name: str) -> dict:
    """Create or return a saved research project."""
    return create_project_impl(name=name)


@tool
def add_to_project(name: str, paper_id: str, source: str) -> dict:
    """Add one paper to a saved research project."""
    return add_to_project_impl(name=name, paper_id=paper_id, source=source)


@tool
def batch_add_to_project(name: str, papers: list[dict] | str) -> dict:
    """Add multiple papers to a project. papers may be a list or JSON-encoded list."""
    if isinstance(papers, str):
        papers = json.loads(papers)
    return batch_add_to_project_impl(name=name, papers=papers)


@tool
def list_projects() -> list[dict]:
    """List saved research projects with paper counts and paper references."""
    return list_projects_impl()
