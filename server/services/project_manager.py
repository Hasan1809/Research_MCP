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


def create_project(name: str) -> dict:
    """Create a new project manifest. Raises ValueError if name is invalid."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Invalid project name: {name!r}")
    path = _project_path(slug)
    if os.path.exists(path):
        logger.info("Project %r already exists — returning existing manifest", slug)
        return _load_manifest(path)
    manifest = {
        "name": slug,
        "created": datetime.now().isoformat(),
        "papers": [],
    }
    _save_manifest(manifest)
    logger.info("Created project %r", slug)
    return manifest


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
