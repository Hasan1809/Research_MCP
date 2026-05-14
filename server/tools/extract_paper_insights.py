from config import FULL_TEXT_CHAR_LIMIT
from services.documents.pdf_service import load_cached
from services.extraction.llm_extractor import extract_field, extract_insights
from services.paper_repository import save_insights
from services.retrieval.vector_store import query_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

MAX_FULL_TEXT_CHARS = FULL_TEXT_CHAR_LIMIT

_RETRIEVAL_QUERIES = {
    "methods": "methodology evaluation criteria reporting framework checklist recommendations approach",
    "results": "results and findings in this paper",
    "datasets": "datasets used in this paper",
    "limitations": "limitations discussed in this paper",
    "future_work": "future work proposed in this paper",
}

_CANDIDATE_K = 8
_MARGIN = 0.10
_MIN_CHUNKS = 2
_MAX_CHUNKS = 4


def _get_full_text(cached: dict) -> str:
    sections = cached.get("sections")
    if sections:
        parts = []
        for sec in sections:
            heading = sec.get("heading", "")
            text = sec.get("text", "").strip()
            if heading and text:
                parts.append(f"## {heading}\n\n{text}")
            elif text:
                parts.append(text)
        if parts:
            return "\n\n".join(parts)
    if cached.get("full_text"):
        return cached["full_text"]
    return "\n\n".join(cached.get("chunks", []))


def _select_chunks_for_field(field: str, query: str, paper_id: str, source: str) -> list[dict]:
    candidates = query_chunks(query, paper_id, source, k=_CANDIDATE_K)
    if not candidates:
        return []
    candidate_indices = [c["chunk_index"] for c in candidates]
    candidate_distances = [round(c["distance"], 4) for c in candidates]
    best_dist = candidates[0]["distance"]
    cutoff = best_dist + _MARGIN
    kept = [c for c in candidates if c["distance"] <= cutoff][:_MAX_CHUNKS]
    if len(kept) < _MIN_CHUNKS:
        kept = candidates[:_MIN_CHUNKS]
    kept_indices = [c["chunk_index"] for c in kept]
    logger.info(
        "Field=%r: candidates=%s distances=%s best=%.4f cutoff=%.4f kept=%s",
        field, candidate_indices, candidate_distances, best_dist, cutoff, kept_indices,
    )
    return kept


def _save_insights(paper_id: str, source: str, result: dict):
    save_insights(source, paper_id, {"paper_id": paper_id, "source": source, **result})
    logger.info("Saved insights for data/insights/%s/%s.json", source, paper_id)


def extract_paper_insights_tool(paper_id: str, source: str) -> dict:
    logger.info("Tool invoked: extract_paper_insights paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    cached = load_cached(source, paper_id)
    if not cached:
        error = "Paper not found in cache. Run ingest_paper_tool first."
        log_invocation("extract_paper_insights_tool", arguments, error=error)
        raise FileNotFoundError(error)

    full_text = _get_full_text(cached)

    if len(full_text) <= MAX_FULL_TEXT_CHARS:
        logger.info("Using single-pass full-text extraction: %d chars", len(full_text))
        try:
            result, raw = extract_insights(full_text)
        except Exception as e:
            log_invocation("extract_paper_insights_tool", arguments, error=str(e))
            raise
        logger.debug("Single-pass raw completion: %s", raw)
        _save_insights(paper_id, source, result)
        log_invocation("extract_paper_insights_tool", arguments, output={
            "path": "full_text_single_pass",
            "context_chars": len(full_text),
            "parsed_output": result,
        })
        return result

    logger.info(
        "Full text too large (%d chars) - using per-field chunk retrieval fallback",
        len(full_text),
    )
    result: dict = {}
    invocation_details: dict = {}

    for field, query in _RETRIEVAL_QUERIES.items():
        chunks = _select_chunks_for_field(field, query, paper_id, source)
        if not chunks:
            logger.warning("No chunks retrieved for field=%r - skipping LLM call", field)
            result[field] = []
            invocation_details[field] = {"chunk_indices": [], "raw_completion": ""}
            continue

        text = "\n".join(c["text"] for c in chunks)
        try:
            field_result, raw = extract_field(field, text)
        except Exception as e:
            log_invocation("extract_paper_insights_tool", arguments, error=str(e))
            raise

        logger.debug("Field=%r raw completion: %s", field, raw)
        result[field] = field_result
        invocation_details[field] = {
            "chunk_indices": [c["chunk_index"] for c in chunks],
            "raw_completion": raw,
        }

    if not result:
        error = "No indexed chunks found. Run index_paper_tool first."
        log_invocation("extract_paper_insights_tool", arguments, error=error)
        raise ValueError(error)

    if not any([
        result.get("methods"),
        result.get("results"),
        result.get("limitations"),
        result.get("future_work"),
    ]):
        error = (
            f"Insight extraction produced empty output for paper_id={paper_id!r}. "
            "The paper must be indexed before fallback retrieval will work. "
            "Run index_paper_tool first, then retry."
        )
        log_invocation("extract_paper_insights_tool", arguments, error=error)
        raise ValueError(error)

    _save_insights(paper_id, source, result)
    log_invocation("extract_paper_insights_tool", arguments, output={
        "path": "per_field_chunk_retrieval",
        "per_field": invocation_details,
        "parsed_output": result,
    })
    return result
