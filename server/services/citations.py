"""Citation metadata normalization and bibliography generation."""
import re
from datetime import datetime
from typing import Any

from config import DATA_DIR
from services.paper_repository import (
    load_paper_cache,
    load_paper_metadata,
    save_paper_metadata,
)
from services.project_manager import get_project_papers
from utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_FORMATS = {"bibtex", "markdown", "ieee", "plaintext"}
_ARTIFACTS_DIR = DATA_DIR / "artifacts" / "bibliographies"


def _looks_bad_title(title: str) -> bool:
    value = (title or "").strip()
    if not value:
        return True
    lowered = value.lower()
    if lowered.startswith("arxiv:"):
        return True
    if len(value.split()) < 4:
        return True
    if value.count(",") >= 2:
        return True
    if value.endswith((" of the", " against", " for", " and")):
        return True
    uppercase_letters = sum(1 for char in value if char.isupper())
    letters = sum(1 for char in value if char.isalpha())
    return bool(letters and uppercase_letters / letters > 0.75 and len(value) > 20)


def _title_from_full_text(cache: dict) -> str:
    text = cache.get("full_text") or ""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()[:12] if line.strip()]
    title_lines = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("published as"):
            continue
        if lower.startswith("arxiv:"):
            continue
        if lower.startswith("abstract") or lower.startswith("arxiv:"):
            break
        if "@" in line or "university" in lower or "institute" in lower or line.count(",") >= 2:
            break
        if re.search(r"^[ivx]+\.", lower):
            break
        title_lines.append(line)
        if len(title_lines) >= 4:
            break
    title = " ".join(title_lines).strip(" -")
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
    title = title.replace("RUNTIMESECURITY", "Runtime Security")
    title = title.replace("FORTOOL", "for Tool")
    title = title.replace("AUGMENTEDLLM", "Augmented LLM")
    title = title.replace("AGENTSAGAINST", "Agents Against")
    title = title.replace("INDIRECTPROMPTINJECTION", "Indirect Prompt Injection")
    title = title.replace("FRAMEWORK", "Framework")
    title = title.replace("AgainstIndirect", "Against Indirect")
    title = title.replace("LLM AGENTSAGAINST", "LLM Agents Against")
    title = title.replace("IPIG UARD", "IPIGuard")
    title = title.title() if title.isupper() else title
    return title


def _metadata_quality(metadata: dict) -> tuple[str, list[str]]:
    warnings = []
    if _looks_bad_title(metadata.get("title", "")):
        warnings.append("suspicious_title")
    if not metadata.get("year"):
        warnings.append("missing_year")
    if not metadata.get("authors"):
        warnings.append("missing_authors")
    if not warnings:
        return "clean", []
    if "suspicious_title" in warnings or "missing_year" in warnings:
        return "fallback", warnings
    return "partial", warnings


def _first_present(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_authors(authors: Any) -> list[str]:
    if not authors:
        return []
    if isinstance(authors, str):
        return [a.strip() for a in authors.split(",") if a.strip()]
    normalized = []
    for author in authors:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = str(author.get("name") or "").strip()
        else:
            name = str(author).strip()
        if name:
            normalized.append(name)
    return normalized


def normalize_paper_metadata(source: str, paper_id: str, raw: dict | None = None) -> dict:
    """Return a stable metadata record for one paper."""
    raw = raw or {}
    existing = load_paper_metadata(source, paper_id) or {}
    cache = load_paper_cache(source, paper_id) or {}
    cache_meta = cache.get("metadata") or {}
    s2_meta = cache.get("semantic_scholar") or raw.get("semantic_scholar") or {}
    external_ids = raw.get("external_ids") or raw.get("externalIds") or s2_meta.get("external_ids") or {}

    semantic_scholar_id = _first_present(
        raw.get("semantic_scholar_id"),
        s2_meta.get("semantic_scholar_id"),
        s2_meta.get("paper_id"),
        paper_id if source == "semantic_scholar" else "",
        existing.get("semantic_scholar_id"),
    )
    arxiv_id = _first_present(
        raw.get("arxiv_id"),
        external_ids.get("ArXiv") if isinstance(external_ids, dict) else "",
        s2_meta.get("arxiv_id"),
        paper_id if source == "arxiv" else "",
        existing.get("arxiv_id"),
    )

    title_candidates = [
        raw.get("title"),
        s2_meta.get("title") if isinstance(s2_meta, dict) else "",
        existing.get("title"),
        cache_meta.get("title"),
        _title_from_full_text(cache),
    ]
    title = ""
    for candidate in title_candidates:
        candidate = str(candidate or "").strip()
        if candidate and not _looks_bad_title(candidate):
            title = candidate
            break
    if not title:
        title = _first_present(*title_candidates)

    authors = _normalize_authors(
        raw.get("authors")
        or (s2_meta.get("authors") if isinstance(s2_meta, dict) else None)
        or existing.get("authors")
        or cache_meta.get("authors")
    )

    metadata = {
        "paper_id": paper_id,
        "source": source,
        "title": title,
        "authors": authors,
        "year": _first_present(
            raw.get("year"),
            s2_meta.get("year") if isinstance(s2_meta, dict) else "",
            existing.get("year"),
            cache_meta.get("year"),
        ),
        "venue": _first_present(
            raw.get("venue"),
            raw.get("publication_venue"),
            raw.get("publicationVenue"),
            existing.get("venue"),
        ),
        "doi": _first_present(
            raw.get("doi"),
            external_ids.get("DOI") if isinstance(external_ids, dict) else "",
            existing.get("doi"),
        ),
        "arxiv_id": arxiv_id,
        "semantic_scholar_id": semantic_scholar_id,
        "url": _first_present(
            raw.get("url"),
            raw.get("semantic_scholar_url"),
            existing.get("url"),
            raw.get("pdf_url"),
            cache.get("pdf_url"),
        ),
        "pdf_url": _first_present(raw.get("pdf_url"), cache.get("pdf_url"), existing.get("pdf_url")),
        "source_database": source,
        "updated_at": datetime.now().isoformat(),
    }
    quality, warnings = _metadata_quality(metadata)
    metadata["metadata_quality"] = quality
    metadata["metadata_warnings"] = warnings
    return metadata


def save_normalized_metadata(source: str, paper_id: str, raw: dict | None = None) -> dict:
    metadata = normalize_paper_metadata(source, paper_id, raw)
    save_paper_metadata(source, paper_id, metadata)
    logger.info("Saved citation metadata: source=%r paper_id=%r", source, paper_id)
    return metadata


def save_search_metadata(papers: list[dict]) -> None:
    for paper in papers:
        paper_id = paper.get("paper_id")
        source = paper.get("source")
        if paper_id and source:
            save_normalized_metadata(source, paper_id, paper)


def _bibtex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def _citation_key(metadata: dict) -> str:
    first_author = "unknown"
    authors = metadata.get("authors") or []
    if authors:
        first_author = authors[0].split()[-1]
    year = metadata.get("year") or "nd"
    token = metadata.get("arxiv_id") or metadata.get("semantic_scholar_id") or metadata.get("paper_id", "")
    base = f"{first_author}{year}{token[:8]}"
    return re.sub(r"[^A-Za-z0-9_:-]", "", base) or "paper"


def _format_bibtex(metadata: dict) -> str:
    fields = {
        "title": metadata.get("title", ""),
        "author": " and ".join(metadata.get("authors") or []),
        "year": metadata.get("year", ""),
        "journal": metadata.get("venue", ""),
        "doi": metadata.get("doi", ""),
        "url": metadata.get("url", ""),
        "archivePrefix": "arXiv" if metadata.get("arxiv_id") else "",
        "eprint": metadata.get("arxiv_id", ""),
    }
    lines = [f"@article{{{_citation_key(metadata)},"]
    for key, value in fields.items():
        if value:
            lines.append(f"  {key} = {{{_bibtex_escape(str(value))}}},")
    lines.append("}")
    return "\n".join(lines)


def _format_markdown(index: int, metadata: dict) -> str:
    authors = ", ".join(metadata.get("authors") or ["Unknown author"])
    year = metadata.get("year") or "n.d."
    title = metadata.get("title") or metadata.get("paper_id")
    venue = metadata.get("venue")
    url = metadata.get("url") or metadata.get("pdf_url")
    tail = f" {venue}." if venue else ""
    if url:
        tail += f" {url}"
    return f"{index}. {authors} ({year}). {title}.{tail}"


def _format_ieee(index: int, metadata: dict) -> str:
    authors = ", ".join(metadata.get("authors") or ["Unknown author"])
    title = metadata.get("title") or metadata.get("paper_id")
    venue = metadata.get("venue")
    year = metadata.get("year")
    parts = [f"[{index}] {authors}, \"{title}\""]
    if venue:
        parts.append(venue)
    if year:
        parts.append(str(year))
    text = ", ".join(parts) + "."
    url = metadata.get("url") or metadata.get("pdf_url")
    if url:
        text += f" Available: {url}"
    return text


def _paper_refs(
    project_name: str | None = None,
    papers: list[dict] | None = None,
    paper_ids: list[str] | None = None,
    source: str = "arxiv",
) -> list[dict]:
    if project_name:
        return get_project_papers(project_name)
    if papers:
        return papers
    if paper_ids:
        return [{"paper_id": paper_id, "source": source} for paper_id in paper_ids]
    raise ValueError("Provide project_name, papers, or paper_ids.")


def generate_bibliography(
    project_name: str | None = None,
    papers: list[dict] | None = None,
    paper_ids: list[str] | None = None,
    source: str = "arxiv",
    format: str = "bibtex",
    save: bool = True,
) -> dict:
    fmt = format.lower()
    if fmt == "plain":
        fmt = "plaintext"
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported bibliography format: {format!r}.")

    refs = _paper_refs(project_name, papers, paper_ids, source)
    included = []
    skipped = []
    entries = []

    for ref in refs:
        paper_id = ref.get("paper_id")
        ref_source = ref.get("source", source)
        metadata = load_paper_metadata(ref_source, paper_id) if paper_id else None
        if paper_id:
            try:
                refreshed = normalize_paper_metadata(ref_source, paper_id, metadata or {})
            except TypeError:
                refreshed = normalize_paper_metadata(ref_source, paper_id)
            if metadata is None or refreshed.get("metadata_quality") == "clean" or _looks_bad_title(metadata.get("title", "")):
                metadata = refreshed
                save_paper_metadata(ref_source, paper_id, metadata)
        if not paper_id or not metadata or not metadata.get("title"):
            skipped.append({
                "paper_id": paper_id or "",
                "source": ref_source,
                "reason": "missing metadata",
            })
            continue
        included.append(metadata)
        if fmt == "bibtex":
            entries.append(_format_bibtex(metadata))
        elif fmt == "markdown":
            entries.append(_format_markdown(len(entries) + 1, metadata))
        else:
            entries.append(_format_ieee(len(entries) + 1, metadata))

    bibliography = "\n\n".join(entries)
    artifact_path = ""
    if save:
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "-", project_name or "papers").strip("-") or "papers"
        extension = "bib" if fmt == "bibtex" else "md"
        path = _ARTIFACTS_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
        path.write_text(bibliography, encoding="utf-8")
        artifact_path = str(path)
        logger.info("Saved bibliography export: %s", path)

    logger.info(
        "Bibliography generated: format=%s included=%d skipped=%d",
        fmt, len(included), len(skipped),
    )
    return {
        "bibliography": bibliography,
        "format": fmt,
        "included": included,
        "skipped": skipped,
        "artifact_path": artifact_path,
    }
