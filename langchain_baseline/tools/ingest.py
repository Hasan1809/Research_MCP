from langchain_core.tools import tool

from langchain_baseline.services import ingest_paper_impl


@tool
def ingest_paper(paper_id: str, source: str = "arxiv") -> dict:
    """Download and cache a paper PDF. Requires an arxiv paper_id and source."""
    return ingest_paper_impl(paper_id, source)
