import json
import os
from services.extraction.llm_extractor import extract_field
from services.retrieval.vector_store import query_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_INSIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "insights")


def _save_insights(paper_id: str, source: str, result: dict):
    folder = os.path.join(_INSIGHTS_DIR, source)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{paper_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"paper_id": paper_id, "source": source, **result}, f, indent=2)
    logger.info("Saved insights to %s", path)

_RETRIEVAL_QUERIES = {
    "methods":      "methodology evaluation criteria reporting framework checklist recommendations approach",
    "results":      "results and findings in this paper",
    "datasets":     "datasets used in this paper",
    "limitations":  "limitations discussed in this paper",
    "future_work":  "future work proposed in this paper",
}

_CANDIDATE_K = 8
_MARGIN = 0.10   # cosine distance margin above best result (cosine range [0, 2])
_MIN_CHUNKS = 2
_MAX_CHUNKS = 4


def _select_chunks_for_field(field: str, query: str, paper_id: str, source: str) -> list[dict]:
    candidates = query_chunks(query, paper_id, source, k=_CANDIDATE_K)
    if not candidates:
        return []

    candidate_indices = [c["chunk_index"] for c in candidates]
    candidate_distances = [round(c["distance"], 4) for c in candidates]
    best_dist = candidates[0]["distance"]
    cutoff = best_dist + _MARGIN

    kept = [c for c in candidates if c["distance"] <= cutoff]
    kept = kept[:_MAX_CHUNKS]
    if len(kept) < _MIN_CHUNKS:
        kept = candidates[:_MIN_CHUNKS]

    kept_indices = [c["chunk_index"] for c in kept]
    logger.info(
        "Field=%r: candidates=%s distances=%s best=%.4f cutoff=%.4f kept=%s",
        field, candidate_indices, candidate_distances, best_dist, cutoff, kept_indices,
    )
    return kept


def extract_paper_insights_tool(paper_id: str, source: str) -> dict:
    logger.info("Tool invoked: extract_paper_insights paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    result = {}
    invocation_details = {}

    for field, query in _RETRIEVAL_QUERIES.items():
        chunks = _select_chunks_for_field(field, query, paper_id, source)

        if not chunks:
            logger.warning("No chunks retrieved for field=%r — skipping LLM call", field)
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

    if not any(result.values()) and not invocation_details:
        error = "No indexed chunks found. Run index_paper_tool first."
        log_invocation("extract_paper_insights_tool", arguments, error=error)
        raise ValueError(error)

    _save_insights(paper_id, source, result)
    log_invocation("extract_paper_insights_tool", arguments, output={
        "per_field": invocation_details,
        "parsed_output": result,
    })
    return result
