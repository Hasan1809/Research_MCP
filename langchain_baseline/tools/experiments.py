from langchain_core.tools import tool

from langchain_baseline.services import suggest_experiments_impl


@tool
def suggest_research_experiments(
    papers: list[dict] | str | None = None,
    project: str = "",
    gap_analysis: dict | None = None,
    compact: bool = True,
) -> dict:
    """
    Suggest concrete research experiments from an existing project or explicit paper list.

    Prefer passing project when papers are already grouped in a saved project; the tool
    will load project papers internally. Do not pass both project and papers unless
    necessary. In compact mode, it reuses cached gap analysis when available and
    generates experiments from gaps plus minimal paper metadata instead of full profiles.

    Preferred:
      suggest_research_experiments(project="my-project", compact=True)

    One-off paper-list mode:
      suggest_research_experiments(papers=[
        {"paper_id": "2602.07652", "source": "arxiv"},
        {"paper_id": "2603.17419", "source": "arxiv"}
      ], compact=True)

    Returns 3-5 experiment proposals with hypotheses, methods,
    baselines, datasets, feasibility ratings, and compact run metadata.
    """
    return suggest_experiments_impl(
        papers=papers,
        project=project or None,
        gap_analysis=gap_analysis,
        compact=compact,
    )
