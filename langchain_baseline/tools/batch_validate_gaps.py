from langchain_core.tools import tool

from langchain_baseline.services import batch_validate_gaps_impl


@tool
def batch_validate_research_gaps(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
) -> dict:
    """
    Validate all candidate research and methodological gaps for a project.

    This performs targeted follow-up academic search for each gap, saves
    individual validation artifacts plus a compact batch summary, and makes the
    validated/refined gaps available to suggest_research_experiments.
    """
    return batch_validate_gaps_impl(
        project=project,
        max_results_per_gap=max_results_per_gap,
        mode=mode,
        max_workers=max_workers,
    )
