from langchain_core.tools import tool

from langchain_baseline.services import ingest_paper_impl


@tool
def ingest_paper(paper_id: str, source: str = "arxiv") -> dict:
    """
    Download and process a paper PDF.

    For source='arxiv', paper_id must be a bare arXiv ID like '2602.07652'.
    For source='semantic_scholar', paper_id must be the Semantic Scholar paperId
    returned by search_papers. Never pass a URL, filename, or placeholder as
    paper_id.

    Example: ingest_paper(paper_id='2602.07652', source='arxiv')

    Caches the result so subsequent calls with the same paper_id are instant.
    Must be called before profile_paper.
    """
    return ingest_paper_impl(paper_id, source)
