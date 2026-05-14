from langchain_core.tools import tool

from langchain_baseline.services import batch_ingest_papers_impl


@tool
def batch_ingest_papers(papers: list[dict] | str) -> dict:
    """Ingest multiple papers concurrently. papers may be a list or
    a JSON-encoded list of {paper_id, source} objects. source may be
    'arxiv' or 'semantic_scholar'."""
    if isinstance(papers, str):
        import json
        papers = json.loads(papers)
    return batch_ingest_papers_impl(papers=papers)
