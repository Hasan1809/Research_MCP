import json
import os
from services.extraction.llm_extractor import build_profile
from services.retrieval.vector_store import query_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "profiles")

# Broad queries to pull diverse, representative chunks from the paper
_PROFILE_QUERIES = [
    "research problem motivation background introduction",
    "main contribution proposed approach methodology",
    "results findings conclusions discussion",
]
_CANDIDATE_K = 6
_MAX_TOTAL_CHUNKS = 15


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
    # Sort by chunk_index so the text reads in document order
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


def build_paper_profile_tool(paper_id: str, source: str) -> dict:
    logger.info("Tool invoked: build_paper_profile paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    chunks = _retrieve_profile_chunks(paper_id, source)
    if not chunks:
        error = "No indexed chunks found. Run index_paper_tool first."
        log_invocation("build_paper_profile_tool", arguments, error=error)
        raise ValueError(error)

    chunk_indices = [c["chunk_index"] for c in chunks]
    # Low chunk indices are likely abstract/intro; high indices likely conclusion
    early = [i for i in chunk_indices if i <= 3]
    late  = [i for i in chunk_indices if i >= max(chunk_indices, default=0) - 3]
    logger.info(
        "Retrieved %d chunks for profile: indices=%s (early/intro=%s, late/conclusion=%s)",
        len(chunks), chunk_indices, early, late,
    )

    text = "\n\n".join(c["text"] for c in chunks)

    logger.info("Generating paper profile...")
    try:
        profile, raw = build_profile(text)
    except Exception as e:
        log_invocation("build_paper_profile_tool", arguments, error=str(e))
        raise

    result = {"paper_id": paper_id, "source": source, **profile}

    folder = os.path.join(_PROFILES_DIR, source)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{paper_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Profile saved to %s", path)

    log_invocation("build_paper_profile_tool", arguments, output={
        "chunk_indices": chunk_indices,
        "profile": result,
    })
    return result
