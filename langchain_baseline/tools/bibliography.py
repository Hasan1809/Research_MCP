import json

from langchain_core.tools import tool

from langchain_baseline.services import generate_bibliography_impl


@tool
def generate_bibliography(
    project_name: str = "",
    papers: list[dict] | str | None = None,
    paper_ids: list[str] | str | None = None,
    source: str = "arxiv",
    format: str = "bibtex",
    save: bool = True,
) -> dict:
    """Generate BibTeX, Markdown, IEEE, or plaintext references from project or paper refs."""
    if isinstance(papers, str):
        papers = json.loads(papers)
    if isinstance(paper_ids, str):
        try:
            paper_ids = json.loads(paper_ids)
        except json.JSONDecodeError:
            paper_ids = [part.strip() for part in paper_ids.split(",") if part.strip()]
    return generate_bibliography_impl(
        project_name=project_name,
        papers=papers,
        paper_ids=paper_ids,
        source=source,
        format=format,
        save=save,
    )
