from langchain_core.tools import tool

from langchain_baseline.services import batch_ingest_papers_impl


def _coerce_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        return int(text)
    return int(value)


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


@tool
def batch_ingest_papers(
    papers: list[dict] | str,
    max_workers: int | str | None = None,
    allow_large_batch: bool | str = False,
) -> dict:
    """Ingest multiple papers concurrently. papers may be a list or
    a JSON-encoded list of {paper_id, source} objects. source may be
    'arxiv' or 'semantic_scholar'."""
    if isinstance(papers, str):
        import json
        papers = json.loads(papers)
    return batch_ingest_papers_impl(
        papers=papers,
        max_workers=_coerce_optional_int(max_workers),
        allow_large_batch=_coerce_bool(allow_large_batch),
    )
