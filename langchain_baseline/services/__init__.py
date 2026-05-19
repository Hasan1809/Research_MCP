import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from config import (
    BATCH_BUILD_PROFILES_MAX_WORKERS,
    BATCH_INGEST_MAX_WORKERS,
    DATA_DIR,
    FULL_TEXT_CHAR_LIMIT,
    IONOS_MODEL,
    LLM_TEMPERATURE,
)
from services.analysis.experiment_suggester import suggest_experiments_for_papers
from services.analysis.gap_validator import batch_validate_gaps, validate_gap
from services.analysis.gap_detector import detect_gaps
from services.citations import (
    generate_bibliography as generate_bibliography_service,
    save_normalized_metadata,
    save_search_metadata,
)
from services.documents.chunking import chunk_sections, chunk_text
from services.documents.pdf_service import detect_sections_from_text, download_and_extract_text, load_cached, save_cached
from services.extraction.llm_extractor import build_profile
from services.paper_repository import load_profile, load_profile_or_insights, save_profile
from services.project_manager import (
    add_paper_to_project,
    batch_add_papers_to_project,
    create_project,
    get_project_papers,
    list_projects,
)
from services.retrieval.aggregator import fetch_papers
from services.retrieval.semantic_scholar_service import resolve_pdf_url
from services.retrieval.vector_store import query_chunks
from langchain_baseline.utils.logger import get_logger, get_session_dir, init_logging, log_invocation
from langchain_baseline.utils.usage_tracker import log_usage

logger = get_logger(__name__)

_ANALYSIS_DIR = DATA_DIR / "analysis"
_MAX_FULL_TEXT_CHARS = FULL_TEXT_CHAR_LIMIT

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


def _paper_cache_path(source: str, paper_id: str) -> str:
    return str(DATA_DIR / "papers" / source / f"{paper_id}.json")


def _log_ingest_summary(
    source: str,
    paper_id: str,
    text_length: int,
    page_count: int,
    section_count: int,
    chunk_count: int,
    section_chunk_count: int,
) -> None:
    logger.info(
        "LangChain ingest summary: source=%s paper_id=%s text_chars=%d pages=%d "
        "sections=%d flat_chunks=%d section_chunks=%d cache_path=%s",
        source,
        paper_id,
        text_length,
        page_count,
        section_count,
        chunk_count,
        section_chunk_count,
        _paper_cache_path(source, paper_id),
    )


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


def _normalize_paper_ids_input(paper_ids: Any) -> list[str] | None:
    if paper_ids is None:
        return None
    if isinstance(paper_ids, str):
        try:
            parsed = json.loads(paper_ids)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in paper_ids.split(",") if part.strip()]
        paper_ids = parsed
    if not isinstance(paper_ids, list):
        raise ValueError("paper_ids must be a list, a JSON-encoded list, or a comma-separated string.")
    return [str(paper_id) for paper_id in paper_ids if str(paper_id).strip()]


def search_papers_impl(query: str, limit: int) -> list[dict]:
    logger.info("LangChain baseline search: query=%r limit=%d", query, limit)
    result = fetch_papers(query, limit)
    save_search_metadata(result)
    log_invocation("lc_search_papers", {"query": query, "limit": limit}, output={
        "result_count": len(result),
    })
    return result


def ingest_paper_impl(paper_id: str, source: str) -> dict:
    logger.info("LangChain baseline ingest: paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source}

    if source not in {"arxiv", "semantic_scholar"}:
        error = f"Unsupported source: {source}. Use 'arxiv' or 'semantic_scholar'."
        log_invocation("lc_ingest_paper", arguments, error=error)
        raise ValueError(error)

    cached = load_cached(source, paper_id)
    if cached is not None:
        metadata = cached.get("metadata", {})
        _log_ingest_summary(
            source,
            paper_id,
            cached.get("text_length", 0),
            metadata.get("page_count", 0),
            len(cached.get("sections", [])),
            cached.get("chunk_count", len(cached.get("chunks", []))),
            len(cached.get("section_chunks", [])),
        )
        result = {
            "paper_id": paper_id,
            "source": source,
            "status": "cached",
            "text_length": cached.get("text_length", 0),
            "chunk_count": cached.get("chunk_count", len(cached.get("chunks", []))),
        }
        log_invocation("lc_ingest_paper", arguments, output=result)
        return result

    metadata = {}
    if source == "arxiv":
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    else:
        pdf_url, metadata = resolve_pdf_url(paper_id)
    text = download_and_extract_text(pdf_url)
    chunks = chunk_text(text)
    structured = detect_sections_from_text(text)
    sections = structured.get("sections", [])
    section_chunks = chunk_sections(sections) if sections else []
    structured_metadata = structured.get("metadata", {})

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
        "metadata": structured_metadata,
    }
    if metadata:
        cached_result["semantic_scholar"] = metadata
    save_cached(source, paper_id, cached_result)
    save_normalized_metadata(source, paper_id, {**metadata, "pdf_url": pdf_url})
    _log_ingest_summary(
        source,
        paper_id,
        len(text),
        structured_metadata.get("page_count", 0),
        len(sections),
        len(chunks),
        len(section_chunks),
    )

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


def batch_ingest_papers_impl(papers: list[dict], max_workers: int | None = None) -> dict:
    """Ingest multiple papers concurrently."""
    succeeded = []
    failed = {}

    if not papers:
        log_invocation("lc_batch_ingest_papers", {"papers": papers}, output={
            "success_count": 0,
            "error_count": 0,
            "succeeded": succeeded,
            "failed": failed,
        })
        return {"succeeded": succeeded, "failed": failed}

    if max_workers is None:
        max_workers = BATCH_INGEST_MAX_WORKERS
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    max_workers = min(len(papers), max_workers)
    logger.info("LangChain batch ingest started: papers=%d max_workers=%d", len(papers), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(ingest_paper_impl, ref["paper_id"], ref["source"]): ref["paper_id"]
            for ref in papers
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
                succeeded.append(paper_id)
                logger.info("Ingest complete: paper_id=%r", paper_id)
            except Exception as e:
                failed[paper_id] = str(e)
                logger.error("Ingest failed: paper_id=%r error=%s", paper_id, e)

    logger.info(
        "LangChain batch ingest complete: success_count=%d failure_count=%d",
        len(succeeded),
        len(failed),
    )
    log_invocation("lc_batch_ingest_papers", {"papers": papers}, output={
        "success_count": len(succeeded),
        "error_count": len(failed),
        "succeeded": succeeded,
        "failed": failed,
    })
    return {"succeeded": succeeded, "failed": failed}


def build_paper_profile_impl(paper_id: str, source: str, force: bool = False) -> dict:
    logger.info("LangChain baseline profile: paper_id=%r source=%r", paper_id, source)
    arguments = {"paper_id": paper_id, "source": source, "force": force}

    if not force:
        cached_profile = load_profile(source, paper_id)
        if cached_profile is not None:
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
    result["_meta"] = {
        "context_path": path_used,
        "context_chars": len(context_text),
        "temperature": LLM_TEMPERATURE,
        "model": IONOS_MODEL,
        "profiled_at": datetime.now().isoformat(),
    }

    save_profile(source, paper_id, result)

    log_invocation("lc_build_paper_profile", arguments, output={
        "path_used": path_used,
        "context_chars": len(context_text),
        "chunk_indices": chunk_indices,
        "paper_id": paper_id,
    })
    return result


def batch_build_profiles_impl(
    papers: list[dict],
    force: bool = False,
    max_workers: int | None = None,
) -> dict:
    """Build profiles for multiple papers concurrently."""
    profiles = {}
    failed = {}

    if not papers:
        log_invocation("lc_batch_build_profiles", {"papers": papers, "force": force}, output={
            "success_count": 0,
            "error_count": 0,
            "succeeded": [],
            "failed": [],
        })
        return {"profiles": profiles, "failed": failed}

    if max_workers is None:
        max_workers = BATCH_BUILD_PROFILES_MAX_WORKERS
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    max_workers = min(len(papers), max_workers)
    logger.info(
        "LangChain batch profile build started: papers=%d max_workers=%d force=%s",
        len(papers),
        max_workers,
        force,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                build_paper_profile_impl,
                ref["paper_id"],
                ref["source"],
                force,
            ): ref["paper_id"]
            for ref in papers
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                profiles[paper_id] = future.result()
                logger.info("Profile complete: paper_id=%r", paper_id)
            except Exception as e:
                failed[paper_id] = str(e)
                logger.error("Profile failed: paper_id=%r error=%s", paper_id, e)

    logger.info(
        "LangChain batch profile build complete: success_count=%d failure_count=%d",
        len(profiles),
        len(failed),
    )
    log_invocation("lc_batch_build_profiles", {"papers": papers, "force": force}, output={
        "success_count": len(profiles),
        "error_count": len(failed),
        "succeeded": list(profiles.keys()),
        "failed": list(failed.keys()),
    })
    return {"profiles": profiles, "failed": failed}


def create_project_impl(name: str) -> dict:
    """Create or return a saved research project."""
    logger.info("LangChain baseline create project: name=%r", name)
    arguments = {"name": name}
    try:
        result = create_project(name)
        log_invocation("lc_create_project", arguments, output={
            "name": result.get("name"),
            "paper_count": len(result.get("papers", [])),
        })
        return result
    except Exception as e:
        log_invocation("lc_create_project", arguments, error=str(e))
        raise


def add_to_project_impl(name: str, paper_id: str, source: str) -> dict:
    """Add one paper to a saved research project."""
    logger.info(
        "LangChain baseline add to project: name=%r paper_id=%r source=%r",
        name,
        paper_id,
        source,
    )
    arguments = {"name": name, "paper_id": paper_id, "source": source}
    try:
        result = add_paper_to_project(name, paper_id, source)
        log_invocation("lc_add_to_project", arguments, output={
            "name": result.get("name"),
            "paper_count": len(result.get("papers", [])),
        })
        return result
    except Exception as e:
        log_invocation("lc_add_to_project", arguments, error=str(e))
        raise


def batch_add_to_project_impl(name: str, papers: list[dict] | str) -> dict:
    """Add multiple papers to a saved research project in one operation."""
    papers = _normalize_papers_input(papers)
    logger.info("LangChain baseline batch add to project: name=%r count=%d", name, len(papers))
    arguments = {"name": name, "papers": papers}
    try:
        result = batch_add_papers_to_project(name, papers)
        log_invocation("lc_batch_add_to_project", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("lc_batch_add_to_project", arguments, error=str(e))
        raise


def list_projects_impl() -> list[dict]:
    """List saved research projects."""
    logger.info("LangChain baseline list projects")
    try:
        result = list_projects()
        log_invocation("lc_list_projects", {}, output={
            "project_count": len(result),
        })
        return result
    except Exception as e:
        log_invocation("lc_list_projects", {}, error=str(e))
        raise


def generate_bibliography_impl(
    project_name: str = "",
    papers: list[dict] | str | None = None,
    paper_ids: list[str] | str | None = None,
    source: str = "arxiv",
    format: str = "bibtex",
    save: bool = True,
) -> dict:
    """Generate a bibliography from a project or explicit paper references."""
    normalized_papers = _normalize_papers_input(papers) if papers else None
    normalized_paper_ids = _normalize_paper_ids_input(paper_ids)
    arguments = {
        "project_name": project_name,
        "papers": normalized_papers,
        "paper_ids": normalized_paper_ids,
        "source": source,
        "format": format,
        "save": save,
    }
    logger.info(
        "LangChain baseline generate bibliography: project=%r format=%r",
        project_name,
        format,
    )
    try:
        result = generate_bibliography_service(
            project_name=project_name or None,
            papers=normalized_papers,
            paper_ids=normalized_paper_ids,
            source=source,
            format=format,
            save=save,
        )
        log_invocation("lc_generate_bibliography", arguments, output={
            "format": result.get("format"),
            "included_count": len(result.get("included", [])),
            "skipped_count": len(result.get("skipped", [])),
            "artifact_path": result.get("artifact_path", ""),
        })
        return result
    except Exception as e:
        log_invocation("lc_generate_bibliography", arguments, error=str(e))
        raise


def detect_gaps_impl(papers: list[dict] | str = None, project: str = None) -> dict:
    if project:
        papers = get_project_papers(project)
        logger.info("LangChain baseline detect gaps from project=%r count=%d", project, len(papers))
    if not papers:
        raise ValueError("detect_research_gaps requires a papers list.")
    papers = _normalize_papers_input(papers)
    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for gap detection.")

    arguments = {"papers": papers, "project": project}
    profiles = [load_profile_or_insights(ref["source"], ref["paper_id"]) for ref in papers]
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


def _find_existing_gap_analysis(paper_ids: list[str]) -> dict | None:
    sorted_ids = sorted(paper_ids)
    for path in sorted(_ANALYSIS_DIR.glob("gap_analysis_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing_ids = sorted(p["paper_id"] for p in data.get("papers", []))
            if existing_ids == sorted_ids:
                return data["analysis"]
        except Exception:
            continue

    for path in sorted(_ANALYSIS_DIR.glob("lc_gap_analysis_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing_ids = sorted(p["paper_id"] for p in data.get("papers", []))
            if existing_ids == sorted_ids:
                return data["analysis"]
        except Exception:
            continue
    return None


def suggest_experiments_impl(
    papers: list[dict] | str = None,
    project: str = None,
    gap_analysis: dict = None,
    compact: bool = True,
) -> dict:
    if project:
        if papers:
            logger.warning(
                "LangChain baseline: both project and papers provided; using project manifest and ignoring papers."
            )
        papers = get_project_papers(project)
        logger.info(
            "LangChain baseline suggest experiments from project=%r active_paper_count=%d compact=%s",
            project,
            len(papers),
            compact,
        )
    if not papers:
        raise ValueError("suggest_research_experiments requires a papers list.")
    papers = _normalize_papers_input(papers)
    if len(papers) < 2:
        raise ValueError("At least 2 papers are required for experiment suggestions.")

    arguments = {
        "project": project,
        "active_paper_count": len(papers),
        "compact": compact,
    } if project else {
        "papers": papers,
        "active_paper_count": len(papers),
        "compact": compact,
    }
    result = suggest_experiments_for_papers(
        papers,
        gap_analysis=gap_analysis,
        compact=compact,
        project=project,
    )

    log_invocation("lc_suggest_experiments", arguments, output={
        "experiment_count": result.get("experiment_count"),
        "gap_count": result.get("gap_count"),
        "gap_source": result.get("gap_source"),
        "project": result.get("project"),
        "active_paper_count": result.get("active_paper_count"),
        "compact": result.get("compact"),
        "gap_analysis_path": result.get("gap_analysis_path"),
        "save_path": result.get("save_path"),
        "error": result.get("error"),
    })
    return result


def validate_gap_impl(
    gap: str,
    project: str = None,
    max_results: int = 10,
    mode: str = "metadata_only",
) -> dict:
    arguments = {
        "gap": gap,
        "project": project,
        "max_results": max_results,
        "mode": mode,
    }
    logger.info(
        "LangChain baseline validate gap: project=%r max_results=%d mode=%s",
        project,
        max_results,
        mode,
    )
    try:
        result = validate_gap(
            gap=gap,
            project=project,
            max_results=max_results,
            mode=mode,
        )
    except Exception as e:
        log_invocation("lc_validate_gap", arguments, error=str(e))
        raise

    log_invocation("lc_validate_gap", arguments, output={
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "results_found": result.get("results_found"),
        "relevant_results": result.get("relevant_results"),
        "artifact_path": result.get("artifact_path"),
    })
    return result


def batch_validate_gaps_impl(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
) -> dict:
    arguments = {
        "project": project,
        "max_results_per_gap": max_results_per_gap,
        "mode": mode,
        "max_workers": max_workers,
    }
    logger.info(
        "LangChain baseline batch validate gaps: project=%r max_results_per_gap=%d mode=%s max_workers=%d",
        project,
        max_results_per_gap,
        mode,
        max_workers,
    )
    try:
        result = batch_validate_gaps(
            project=project,
            max_results_per_gap=max_results_per_gap,
            mode=mode,
            max_workers=max_workers,
        )
    except Exception as e:
        log_invocation("lc_batch_validate_gaps", arguments, error=str(e))
        raise

    log_invocation("lc_batch_validate_gaps", arguments, output={
        "project": result.get("project"),
        "gap_count": result.get("gap_count"),
        "validated_count": result.get("validated_count"),
        "failed_count": result.get("failed_count"),
        "status_counts": result.get("status_counts"),
        "batch_artifact_path": result.get("batch_artifact_path"),
    })
    return result


__all__ = [
    "add_to_project_impl",
    "batch_add_to_project_impl",
    "batch_build_profiles_impl",
    "batch_ingest_papers_impl",
    "batch_validate_gaps_impl",
    "build_paper_profile_impl",
    "create_project_impl",
    "detect_gaps_impl",
    "generate_bibliography_impl",
    "ingest_paper_impl",
    "list_projects_impl",
    "search_papers_impl",
    "suggest_experiments_impl",
    "validate_gap_impl",
    "get_logger",
    "get_session_dir",
    "init_logging",
    "log_usage",
]
