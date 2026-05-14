from langchain_core.tools import tool

from langchain_baseline.services import suggest_experiments_impl


@tool
def suggest_research_experiments(
    papers: list[dict] | str | None = None,
    project: str = "",
    gap_analysis: dict | None = None,
) -> dict:
    """
    Suggest concrete experiments based on research gaps across papers.

    Requires profile_paper to have been called for each paper first.
    Pass papers as a list of dicts or pass project as a saved project name.
    Requires at least 2 papers.
    Always call this after detect_research_gaps, not instead of it.

    Example:
      suggest_research_experiments(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "2603.17419", "source": "arxiv"}
      ])

    Returns 3-5 experiment proposals with hypotheses, methods,
    baselines, datasets, and feasibility ratings.
    """
    return suggest_experiments_impl(
        papers=papers,
        project=project or None,
        gap_analysis=gap_analysis,
    )
