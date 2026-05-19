from datetime import datetime

from config import FULL_TEXT_CHAR_LIMIT, IONOS_MODEL, LLM_TEMPERATURE
from services.documents.pdf_service import load_cached
from services.extraction.llm_extractor import build_profile
from services.paper_repository import load_profile, save_profile
from services.retrieval.vector_store import query_chunks
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

MAX_FULL_TEXT_CHARS = FULL_TEXT_CHAR_LIMIT

_PROFILE_QUERIES = [
    "research problem motivation background introduction",
    "main contribution proposed approach methodology",
    "results findings conclusions discussion",
]
_CANDIDATE_K = 6
_MAX_TOTAL_CHUNKS = 15

_PRIORITY_SECTION_KEYWORDS = {
    "abstract", "introduction", "conclusion", "conclusions",
    "discussion", "method", "methods", "approach", "related work",
}


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


def build_paper_profile_tool(paper_id: str, source: str, force: bool = False) -> dict:
    """
    Generate a comprehensive 13-field profile of a paper using LLM analysis.

    For source='arxiv', paper_id must be a bare arXiv ID like '2602.07652'.
    For source='semantic_scholar', paper_id must be the Semantic Scholar paperId
    returned by search_papers_tool. Never pass a URL, filename, or placeholder
    as paper_id.

    Example: build_paper_profile_tool(paper_id='2602.07652', source='arxiv')
    Example: build_paper_profile_tool(paper_id='649def34f8be52c8b66281af98ae884c09aef38b', source='semantic_scholar')

    Must be called after the paper has been ingested with the same paper_id and source.
    Must be called on at least 2 papers before detect_gaps_tool.
    """
    logger.info("Tool invoked: build_paper_profile paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    if not force:
        cached_profile = load_profile(source, paper_id)
        if cached_profile is not None:
            logger.info("Profile already exists for %r - returning cached", paper_id)
            log_invocation("build_paper_profile_tool", arguments, output={
                "path_used": "cached_profile",
                "profile": cached_profile,
            })
            return cached_profile

    cached = load_cached(source, paper_id)
    if not cached:
        error = "Paper not found in cache. Run batch_ingest_papers_tool first."
        log_invocation("build_paper_profile_tool", arguments, error=error)
        raise FileNotFoundError(error)

    full_text = _get_full_text(cached)
    path_used = (
        "full_text" if len(full_text) <= MAX_FULL_TEXT_CHARS
        else "priority_sections_or_chunk_retrieval"
    )

    if len(full_text) <= MAX_FULL_TEXT_CHARS:
        logger.info(
            "Using full-text path: %d chars (limit=%d)", len(full_text), MAX_FULL_TEXT_CHARS
        )
        context_text = full_text
        chunk_indices = []
    else:
        logger.info(
            "Full text too large (%d chars > %d limit) - trying priority-sections fallback",
            len(full_text), MAX_FULL_TEXT_CHARS,
        )
        sections = cached.get("sections", [])
        priority_sections = [
            sec for sec in sections
            if any(kw in sec.get("heading", "").lower() for kw in _PRIORITY_SECTION_KEYWORDS)
        ]
        priority_text = "\n\n".join(
            f"## {sec['heading']}\n\n{sec['text'].strip()}"
            for sec in priority_sections
            if sec.get("text", "").strip()
        )
        if priority_text and len(priority_text) <= MAX_FULL_TEXT_CHARS:
            logger.info(
                "Priority-sections fallback: %d sections, %d chars",
                len(priority_sections), len(priority_text),
            )
            context_text = priority_text
            path_used = "priority_sections"
            if len(context_text) < 5000:
                logger.warning(
                    "Priority sections too short (%d chars) - using truncated full text",
                    len(context_text),
                )
                context_text = full_text[:MAX_FULL_TEXT_CHARS]
                path_used = "truncated_full_text"
            chunk_indices = []
        else:
            if priority_text:
                logger.info(
                    "Priority sections still too large (%d chars) - falling back to chunk retrieval",
                    len(priority_text),
                )
            else:
                logger.info("No priority sections found - falling back to chunk retrieval")
            chunks = _retrieve_profile_chunks(paper_id, source)
            if not chunks:
                logger.warning(
                    "No indexed chunks found for %s/%s - using truncated full text fallback",
                    source,
                    paper_id,
                )
                context_text = full_text[:MAX_FULL_TEXT_CHARS]
                path_used = "truncated_full_text"
                chunk_indices = []
            else:
                chunk_indices = [c["chunk_index"] for c in chunks]
                early = [i for i in chunk_indices if i <= 3]
                late = [i for i in chunk_indices if i >= max(chunk_indices) - 3]
                logger.info(
                    "Retrieved %d chunks: indices=%s (early=%s late=%s)",
                    len(chunks), chunk_indices, early, late,
                )
                context_text = "\n\n".join(c["text"] for c in chunks)
                path_used = "chunk_retrieval"

    logger.info("Generating paper profile (path=%s)...", path_used)
    try:
        profile, raw = build_profile(context_text, paper_id=paper_id)
    except Exception as e:
        log_invocation("build_paper_profile_tool", arguments, error=str(e))
        raise

    result = {"paper_id": paper_id, "source": source, **profile}
    result["_meta"] = {
        "context_path": path_used,
        "context_chars": len(context_text),
        "temperature": LLM_TEMPERATURE,
        "model": IONOS_MODEL,
        "profiled_at": datetime.now().isoformat(),
    }

    save_profile(source, paper_id, result)
    logger.info("Profile saved for data/profiles/%s/%s.json", source, paper_id)

    log_invocation("build_paper_profile_tool", arguments, output={
        "path_used": path_used,
        "context_chars": len(context_text),
        "chunk_indices": chunk_indices,
        "profile": result,
    })
    return result
