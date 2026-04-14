import json
import os
from services.documents.pdf_service import load_cached
from services.extraction.llm_extractor import build_profile
from services.retrieval.vector_store import query_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "profiles")

MAX_FULL_TEXT_CHARS = int(os.environ.get("FULL_TEXT_CHAR_LIMIT", 80_000))

# Fallback retrieval queries (used only when full text exceeds limit)
_PROFILE_QUERIES = [
    "research problem motivation background introduction",
    "main contribution proposed approach methodology",
    "results findings conclusions discussion",
]
_CANDIDATE_K = 6
_MAX_TOTAL_CHUNKS = 15


def _get_full_text(cached: dict) -> str:
    """Reconstruct coherent paper text from cached data, preserving section headings."""
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
    # Fall back to flat full_text or joining chunks
    if cached.get("full_text"):
        return cached["full_text"]
    return "\n\n".join(cached.get("chunks", []))


def _retrieve_profile_chunks(paper_id: str, source: str) -> list[dict]:
    seen: set[int] = set()
    chunks: list[dict] = []
    for query in _PROFILE_QUERIES:
        for chunk in query_chunks(query, paper_id, source, k=_CANDIDATE_K):
            if chunk["chunk_index"] not in seen:
                seen.add(chunk["chunk_index"])
                chunks.append(chunk)
            if len(chunks) >= _MAX_TOTAL_CHUNKS:
                break
        if len(chunks) >= _MAX_TOTAL_CHUNKS:
            break
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


def build_paper_profile_tool(paper_id: str, source: str) -> dict:
    logger.info("Tool invoked: build_paper_profile paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    cached = load_cached(source, paper_id)
    if not cached:
        error = "Paper not found in cache. Run ingest_paper_tool first."
        log_invocation("build_paper_profile_tool", arguments, error=error)
        raise FileNotFoundError(error)

    full_text = _get_full_text(cached)
    path_used = "full_text" if len(full_text) <= MAX_FULL_TEXT_CHARS else "chunk_retrieval"

    if len(full_text) <= MAX_FULL_TEXT_CHARS:
        logger.info(
            "Using full-text path: %d chars (limit=%d)", len(full_text), MAX_FULL_TEXT_CHARS
        )
        context_text = full_text
        chunk_indices = []
    else:
        logger.info(
            "Full text too large (%d chars > %d limit) — falling back to chunk retrieval",
            len(full_text), MAX_FULL_TEXT_CHARS,
        )
        chunks = _retrieve_profile_chunks(paper_id, source)
        if not chunks:
            error = "No indexed chunks found. Run index_paper_tool first."
            log_invocation("build_paper_profile_tool", arguments, error=error)
            raise ValueError(error)
        chunk_indices = [c["chunk_index"] for c in chunks]
        early = [i for i in chunk_indices if i <= 3]
        late  = [i for i in chunk_indices if i >= max(chunk_indices) - 3]
        logger.info(
            "Retrieved %d chunks: indices=%s (early=%s late=%s)",
            len(chunks), chunk_indices, early, late,
        )
        context_text = "\n\n".join(c["text"] for c in chunks)

    logger.info("Generating paper profile (path=%s)...", path_used)
    try:
        profile, raw = build_profile(context_text)
    except Exception as e:
        log_invocation("build_paper_profile_tool", arguments, error=str(e))
        raise

    result = {"paper_id": paper_id, "source": source, **profile}

    folder = os.path.join(_PROFILES_DIR, source)
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f"{paper_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Profile saved to %s", save_path)

    log_invocation("build_paper_profile_tool", arguments, output={
        "path_used": path_used,
        "context_chars": len(context_text),
        "chunk_indices": chunk_indices,
        "profile": result,
    })
    return result
