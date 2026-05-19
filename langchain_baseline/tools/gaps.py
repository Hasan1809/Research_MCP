from langchain_core.tools import tool

from langchain_baseline.services import detect_gaps_impl


@tool
def detect_research_gaps(papers: list[dict] | str | None = None, project: str = "") -> dict:
    """
    Identify research gaps across multiple profiled papers.

    Requires paper profiles to have been built for each paper first.
    Pass papers as a list of dicts or pass project as a saved project name.
    Requires at least 2 papers.

    Example:
      detect_research_gaps(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "2603.17419", "source": "arxiv"}
      ])

    Returns research_gaps, methodological_gaps, contradictions,
    connections, and field_summary.
    """
    return detect_gaps_impl(papers=papers, project=project or None)
