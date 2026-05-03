from langchain_core.tools import tool

from langchain_baseline.services import detect_gaps_impl


@tool
def detect_research_gaps(papers: list[dict] | str) -> dict:
    """Detect cross-paper research gaps from paper refs. papers may be a list or a JSON-encoded list of {paper_id, source} objects."""
    return detect_gaps_impl(papers=papers)
