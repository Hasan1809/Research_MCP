from langchain_core.tools import tool

from langchain_baseline.services import search_papers_impl


@tool
def search_papers(query: str, limit: int = 5) -> list[dict]:
    """Search academic papers. For niche topics, call multiple related queries."""
    return search_papers_impl(query, limit)
