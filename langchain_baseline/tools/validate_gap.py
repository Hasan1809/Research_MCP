from langchain_core.tools import tool

from langchain_baseline.services import validate_gap_impl


@tool
def validate_research_gap(
    gap: str,
    project: str = "",
    max_results: int = 10,
    mode: str = "metadata_only",
) -> dict:
    """
    Validate a candidate research gap with targeted follow-up academic search.

    Use after detect_research_gaps when a candidate gap needs checking against
    wider literature. This searches existing arXiv and Semantic Scholar services,
    classifies metadata-only evidence, marks project papers versus external
    papers when project is provided, and returns a validation status plus an
    optional refined gap. It does not prove a gap absolutely; it classifies the
    available evidence.
    """
    return validate_gap_impl(
        gap=gap,
        project=project or None,
        max_results=max_results,
        mode=mode,
    )
