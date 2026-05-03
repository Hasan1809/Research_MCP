import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT_DIR / "server"
SERVER_SERVICES_DIR = SERVER_DIR / "services"
BASELINE_UTILS_DIR = ROOT_DIR / "langchain_baseline" / "utils"

load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("USAGE_TOOL_PREFIX", "lc_")

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def _ensure_namespace_package(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None) == [str(path)]:
        return

    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_namespace_package("services", SERVER_SERVICES_DIR)
_ensure_namespace_package("utils", BASELINE_UTILS_DIR)

from services.analysis.experiment_suggester import suggest_experiments
from services.analysis.gap_detector import detect_gaps
from services.documents.chunking import chunk_sections, chunk_text
from services.documents.pdf_service import detect_sections_from_text, download_and_extract_text, load_cached, save_cached
from services.extraction.llm_extractor import build_profile
from services.retrieval.aggregator import fetch_papers
from services.retrieval.vector_store import query_chunks
from langchain_baseline.utils.logger import get_logger, get_session_dir, init_logging, log_invocation
from langchain_baseline.utils.usage_tracker import log_usage

logger = get_logger(__name__)

_PROFILES_DIR = ROOT_DIR / "data" / "profiles"
_ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
_INSIGHTS_DIR = ROOT_DIR / "data" / "insights"
_MAX_FULL_TEXT_CHARS = int(os.environ.get("FULL_TEXT_CHAR_LIMIT", 160_000))

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
    seen = set()
    chunks = []
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


def _load_profile_or_insights(paper_id: str, source: str) -> dict:
    profile_path = _PROFILES_DIR / source / f"{paper_id}.json"
    insights_path = _INSIGHTS_DIR / source / f"{paper_id}.json"

    if profile_path.exists():
        with open(profile_path, encoding="utf-8") as f:
            return json.load(f)
    if insights_path.exists():
        with open(insights_path, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No profile or insights found for paper_id={paper_id!r} source={source!r}."
    )


def _normalize_papers_input(papers: Any) -> list[dict]:
    if isinstance(papers, str):
        try:
            papers = json.loads(papers)
        except json.JSONDecodeError as e:
            raise ValueError("papers must be a list or a JSON-encoded list of paper refs.") from e

    if not isinstance(papers, list):
        raise ValueError("papers must be a list of {'paper_id': ..., 'source': ...} objects.")

    normalized = []
    for item in papers:
        if not isinstance(item, dict):
            raise ValueError("Each papers item must be an object with paper_id and source.")
        paper_id = item.get("paper_id")
        source = item.get("source")
        if not paper_id or not source:
            raise ValueError("Each papers item must contain paper_id and source.")
        normalized.append({"paper_id": str(paper_id), "source": str(source)})
    return normalized


def search_papers_impl(query: str, limit: int) -> list[dict]:
    logger.info("LangChain baseline search: query=%r limit=%d", query, limit)
    result = fetch_papers(query, limit)
    log_invocation("lc_search_papers", {"query": query, "limit": limit}, output={
        "result_count": len(result),
    })
    return result


def ingest_paper_impl(paper_id: str, source: str) -> dict:
    logger.info("LangChain baseline ingest: paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    if source != "arxiv":
        error = f"Unsupported source: {source}. Only 'arxiv' is supported."
        log_invocation("lc_ingest_paper", arguments, error=error)
        raise ValueError(error)

    cached = load_cached(source, paper_id)
    if cached is not None:
        result = {
            "paper_id": paper_id,
            "source": source,
            "status": "cached",
            "text_length": cached.get("text_length", 0),
            "chunk_count": cached.get("chunk_count", len(cached.get("chunks", []))),
        }
        log_invocation("lc_ingest_paper", arguments, output=result)
        return result

    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    text = download_and_extract_text(pdf_url)
    chunks = chunk_text(text)
    structured = detect_sections_from_text(text)
    sections = structured.get("sections", [])
    section_chunks = chunk_sections(sections) if sections else []

    cached_result = {
        "paper_id": paper_id,
        "source": source,
        "pdf_url": pdf_url,
        "text_length": len(text),
        "full_text": text,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "sections": sections,
        "section_chunks": section_chunks,
        "metadata": structured.get("metadata", {}),
    }
    save_cached(source, paper_id, cached_result)

    result = {
        "paper_id": paper_id,
        "source": source,
        "status": "ingested",
        "text_length": len(text),
        "chunk_count": len(chunks),
        "section_count": len(sections),
    }
    log_invocation("lc_ingest_paper", arguments, output=result)
    return result


def build_paper_profile_impl(paper_id: str, source: str, force: bool = False) -> dict:
    logger.info("LangChain baseline profile: paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source, "force": force}

    profile_path = _PROFILES_DIR / source / f"{paper_id}.json"
    if profile_path.exists() and not force:
        with open(profile_path, encoding="utf-8") as f:
            cached_profile = json.load(f)
        log_invocation("lc_build_paper_profile", arguments, output={
            "path_used": "cached_profile",
            "paper_id": paper_id,
        })
        return cached_profile

    cached = load_cached(source, paper_id)
    if not cached:
        error = "Paper not found in cache. Run ingest_paper first."
        log_invocation("lc_build_paper_profile", arguments, error=error)
        raise FileNotFoundError(error)

    full_text = _get_full_text(cached)
    path_used = "full_text"
    context_text = full_text
    chunk_indices = []

    if len(full_text) > _MAX_FULL_TEXT_CHARS:
        path_used = "priority_sections_or_chunk_retrieval"
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

        if priority_text and len(priority_text) <= _MAX_FULL_TEXT_CHARS:
            context_text = priority_text
        else:
            chunks = _retrieve_profile_chunks(paper_id, source)
            if chunks:
                chunk_indices = [c["chunk_index"] for c in chunks]
                context_text = "\n\n".join(c["text"] for c in chunks)
                path_used = "chunk_retrieval"
            else:
                context_text = full_text[:_MAX_FULL_TEXT_CHARS]
                path_used = "truncated_full_text"

    profile, _ = build_profile(context_text, paper_id=paper_id)
    result = {"paper_id": paper_id, "source": source, **profile}

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log_invocation("lc_build_paper_profile", arguments, output={
        "path_used": path_used,
        "context_chars": len(context_text),
        "chunk_indices": chunk_indices,
        "paper_id": paper_id,
    })
    return result


def detect_gaps_impl(papers: list[dict] | str = None, project: str = None) -> dict:
    if project:
        raise NotImplementedError("Project-based lookup is not implemented in the LangChain baseline.")
    if not papers:
        raise ValueError("detect_research_gaps requires a papers list.")
    papers = _normalize_papers_input(papers)
    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for gap detection.")

    arguments = {"papers": papers, "project": project}
    profiles = [_load_profile_or_insights(ref["paper_id"], ref["source"]) for ref in papers]
    result, _ = detect_gaps(profiles)

    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(ref.get("paper_id", "?").replace("/", "-") for ref in papers[:3])
    save_path = _ANALYSIS_DIR / f"lc_gap_analysis_{timestamp}_{paper_ids}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"papers": papers, "analysis": result}, f, indent=2)

    log_invocation("lc_detect_gaps", arguments, output={
        "research_gap_count": len(result.get("research_gaps", [])),
        "methodological_gap_count": len(result.get("methodological_gaps", [])),
        "save_path": str(save_path),
    })
    return result


def suggest_experiments_impl(papers: list[dict] | str = None, project: str = None) -> dict:
    if project:
        raise NotImplementedError("Project-based lookup is not implemented in the LangChain baseline.")
    if not papers:
        raise ValueError("suggest_research_experiments requires a papers list.")
    papers = _normalize_papers_input(papers)
    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for experiment suggestions.")

    arguments = {"papers": papers, "project": project}
    profiles = [_load_profile_or_insights(ref["paper_id"], ref["source"]) for ref in papers]
    gap_analysis, _ = detect_gaps(profiles)
    result, _ = suggest_experiments(gap_analysis, profiles)

    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_ids = "_".join(ref.get("paper_id", "?").replace("/", "-") for ref in papers[:3])
    save_path = _ANALYSIS_DIR / f"lc_experiments_{timestamp}_{paper_ids}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"papers": papers, "gaps": gap_analysis, "experiments": result}, f, indent=2)

    experiments = result.get("experiments", [])
    log_invocation("lc_suggest_experiments", arguments, output={
        "experiment_count": len(experiments),
        "save_path": str(save_path),
    })
    return {"gaps": gap_analysis, "experiments": experiments}


__all__ = [
    "build_paper_profile_impl",
    "detect_gaps_impl",
    "ingest_paper_impl",
    "search_papers_impl",
    "suggest_experiments_impl",
    "get_logger",
    "get_session_dir",
    "init_logging",
    "log_usage",
]
