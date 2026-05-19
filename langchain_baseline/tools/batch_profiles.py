from langchain_core.tools import tool

from langchain_baseline.services import batch_build_profiles_impl


@tool
def batch_build_profiles(
    papers: list[dict] | str,
    force: bool = False,
    max_workers: int | None = None,
) -> dict:
    """Build profiles for multiple papers concurrently. papers may be
    a list or a JSON-encoded list of {paper_id, source} objects.
    source may be 'arxiv' or 'semantic_scholar'.
    Each paper must be ingested first."""
    if isinstance(papers, str):
        import json
        papers = json.loads(papers)
    return batch_build_profiles_impl(
        papers=papers,
        force=force,
        max_workers=max_workers,
    )
