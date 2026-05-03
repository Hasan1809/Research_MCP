from langchain_core.tools import tool

from langchain_baseline.services import suggest_experiments_impl


@tool
def suggest_research_experiments(papers: list[dict] | str) -> dict:
    """Suggest experiments from profiled paper refs. papers may be a list or a JSON-encoded list of {paper_id, source} objects."""
    return suggest_experiments_impl(papers=papers)
