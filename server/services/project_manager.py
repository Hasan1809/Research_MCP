"""Project manifest management — groups papers by research topic."""
import json
import os
import re
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "projects")

_SLUG_RE = re.compile(r"[^a-z0-9-]")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.strip().lower()).strip("-")


def _project_path(name: str) -> str:
    return os.path.join(_PROJECTS_DIR, f"{name}.json")


def _load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: dict) -> None:
    os.makedirs(_PROJECTS_DIR, exist_ok=True)
    path = _project_path(manifest["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved project manifest: %s", path)


def create_project(name: str, overwrite: bool = False) -> dict:
    """Create a new project manifest. Raises ValueError if name is invalid."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Invalid project name: {name!r}")
    path = _project_path(slug)
    existed = os.path.exists(path)
    if existed and not overwrite:
        logger.info("Project %r already exists — returning existing manifest", slug)
        manifest = _load_manifest(path)
        manifest["reused_existing"] = True
        return manifest
    manifest = {
        "name": slug,
        "created": datetime.now().isoformat(),
        "papers": [],
        "reused_existing": False,
        "overwritten": bool(existed and overwrite),
    }
    _save_manifest(manifest)
    logger.info("Created project %r", slug)
    return manifest


def clear_project(name: str) -> dict:
    """Clear all papers from an existing project manifest."""
    slug = _slugify(name)
    path = _project_path(slug)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Project {slug!r} not found.")
    manifest = _load_manifest(path)
    removed_count = len(manifest.get("papers", []))
    manifest["papers"] = []
    manifest["cleared_at"] = datetime.now().isoformat()
    _save_manifest(manifest)
    logger.info("Cleared project %r removed=%d", slug, removed_count)
    return {
        "project": slug,
        "removed_count": removed_count,
        "paper_count": 0,
        "manifest": manifest,
    }


def add_paper_to_project(name: str, paper_id: str, source: str) -> dict:
    """Add a paper to an existing project. No-op if already present."""
    slug = _slugify(name)
    path = _project_path(slug)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Project {slug!r} not found. Run create_project first.")
    manifest = _load_manifest(path)
    already = any(
        p["paper_id"] == paper_id and p["source"] == source
        for p in manifest["papers"]
    )
    if already:
        logger.info("Paper %r (%s) already in project %r", paper_id, source, slug)
        return manifest
    manifest["papers"].append({
        "paper_id": paper_id,
        "source": source,
        "added": datetime.now().isoformat(),
    })
    _save_manifest(manifest)
    logger.info("Added paper %r (%s) to project %r", paper_id, source, slug)
    return manifest


def batch_add_papers_to_project(name: str, papers: list[dict]) -> dict:
    """Add multiple papers to a project, saving the manifest once."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Invalid project name: {name!r}")

    path = _project_path(slug)
    if os.path.exists(path):
        manifest = _load_manifest(path)
    else:
        manifest = {
            "name": slug,
            "created": datetime.now().isoformat(),
            "papers": [],
        }
        logger.info("Creating project %r during batch add", slug)

    added = []
    duplicates = []
    skipped = []
    failed = []
    seen = {
        (paper.get("paper_id"), paper.get("source"))
        for paper in manifest.get("papers", [])
    }

    for index, ref in enumerate(papers or []):
        try:
            if not isinstance(ref, dict):
                skipped.append({
                    "index": index,
                    "paper_id": "",
                    "source": "",
                    "reason": "paper reference must be an object",
                })
                continue

            paper_id = str(ref.get("paper_id") or "").strip()
            source = str(ref.get("source") or "").strip()
            if not paper_id or not source:
                skipped.append({
                    "index": index,
                    "paper_id": paper_id,
                    "source": source,
                    "reason": "missing paper_id or source",
                })
                continue

            key = (paper_id, source)
            entry = {"paper_id": paper_id, "source": source}
            if key in seen:
                duplicates.append({**entry, "reason": "already in project"})
                continue

            manifest["papers"].append({
                "paper_id": paper_id,
                "source": source,
                "added": datetime.now().isoformat(),
            })
            seen.add(key)
            added.append(entry)
        except Exception as e:
            failed.append({
                "index": index,
                "paper_id": str(ref.get("paper_id", "")) if isinstance(ref, dict) else "",
                "source": str(ref.get("source", "")) if isinstance(ref, dict) else "",
                "error": str(e),
            })

    if added or not os.path.exists(path):
        _save_manifest(manifest)

    logger.info(
        "Batch add to project %r complete: input=%d added=%d duplicates=%d skipped=%d failed=%d",
        slug,
        len(papers or []),
        len(added),
        len(duplicates),
        len(skipped),
        len(failed),
    )
    return {
        "project": slug,
        "added": added,
        "skipped": skipped,
        "duplicates": duplicates,
        "failed": failed,
        "summary": {
            "input_count": len(papers or []),
            "added_count": len(added),
            "duplicate_count": len(duplicates),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "paper_count": len(manifest.get("papers", [])),
        },
    }


def remove_paper_from_project(name: str, paper_id: str, source: str) -> dict:
    """Remove a paper from a project. No-op if not present."""
    slug = _slugify(name)
    path = _project_path(slug)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Project {slug!r} not found.")
    manifest = _load_manifest(path)
    before = len(manifest["papers"])
    manifest["papers"] = [
        p for p in manifest["papers"]
        if not (p["paper_id"] == paper_id and p["source"] == source)
    ]
    if len(manifest["papers"]) < before:
        _save_manifest(manifest)
        logger.info("Removed paper %r (%s) from project %r", paper_id, source, slug)
    else:
        logger.info("Paper %r (%s) not found in project %r — no change", paper_id, source, slug)
    return manifest


def get_project(name: str) -> dict:
    """Load a project manifest. Raises FileNotFoundError if not found."""
    slug = _slugify(name)
    path = _project_path(slug)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Project {slug!r} not found.")
    return _load_manifest(path)


def list_projects() -> list[dict]:
    """List all projects with name, created date, and paper count."""
    os.makedirs(_PROJECTS_DIR, exist_ok=True)
    results = []
    for fname in sorted(os.listdir(_PROJECTS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            manifest = _load_manifest(os.path.join(_PROJECTS_DIR, fname))
            results.append({
                "name": manifest["name"],
                "created": manifest.get("created", ""),
                "paper_count": len(manifest.get("papers", [])),
                "papers": manifest.get("papers", []),
            })
        except Exception:
            logger.warning("Could not read project file: %s", fname)
    return results


def get_project_papers(name: str) -> list[dict]:
    """Return the list of paper references for a project."""
    return get_project(name)["papers"]
