from langchain_core.tools import tool

from langchain_baseline.services import build_paper_profile_impl


@tool
def profile_paper(paper_id: str, source: str = "arxiv") -> dict:
    """
    Generate a comprehensive 13-field profile of a paper using LLM analysis.

    For source='arxiv', paper_id must be a bare arXiv ID like '2602.07652'.
    For source='semantic_scholar', paper_id must be the Semantic Scholar paperId
    returned by search_papers. Never pass a URL, filename, or placeholder as
    paper_id.

    Example: profile_paper(paper_id='2602.07652', source='arxiv')

    Must be called after ingest_paper with the same paper_id and source.
    Must be called on at least 2 papers before detect_research_gaps.
    """
    return build_paper_profile_impl(paper_id, source)
