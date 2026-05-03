from langchain_core.tools import tool

from langchain_baseline.services import build_paper_profile_impl


@tool
def profile_paper(paper_id: str, source: str = "arxiv") -> dict:
    """Build a paper profile. Run ingest_paper first for the same paper."""
    return build_paper_profile_impl(paper_id, source)
