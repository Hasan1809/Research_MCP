"""MCP tools for managing research projects."""
from services.project_manager import (
    create_project,
    batch_add_papers_to_project,
    clear_project,
)
from services.paper_repository import load_profile
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def create_project_tool(name: str, overwrite: bool = False) -> dict:
    """
    Create a new research project that groups papers by topic.

    Args:
        name: Project name (e.g. "moe-efficiency"). Will be slugified
              to lowercase with hyphens.
        overwrite: If true, replace an existing project with an empty manifest.

    Returns:
        The project manifest dict with name, created, and papers list.
    """
    logger.info("Tool invoked: create_project name=%r overwrite=%s", name, overwrite)
    arguments = {"name": name, "overwrite": overwrite}
    try:
        result = create_project(name, overwrite=overwrite)
        log_invocation("create_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("create_project_tool", arguments, error=str(e))
        raise

def _normalize_source(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if normalized in {"semanticscholar", "semantic-scholar", "semantic scholar"}:
        return "semantic_scholar"
    return normalized


def batch_add_to_project_tool(
    name: str,
    papers: list[dict],
) -> dict:
    """
    Add multiple papers to a project in one operation.

    Args:
        name: Project name. Will be slugified to match project manifests.
        papers: List of {"paper_id": "...", "source": "..."} references.

    Returns:
        Structured result with added, skipped, duplicate, failed, and summary counts.
    """
    logger.info(
        "Tool invoked: batch_add_to_project name=%r count=%d require_profiles=True",
        name,
        len(papers or []),
    )
    arguments = {"name": name, "papers": papers, "require_profiles": True}
    try:
        normalized = []
        skipped_unprofiled = []
        for index, ref in enumerate(papers or []):
            if not isinstance(ref, dict):
                normalized.append(ref)
                continue
            paper_id = str(ref.get("paper_id") or "").strip()
            source = _normalize_source(ref.get("source"))
            clean_ref = {**ref, "paper_id": paper_id, "source": source}
            if paper_id and source and load_profile(source, paper_id) is None:
                skipped_unprofiled.append({
                    "index": index,
                    "paper_id": paper_id,
                    "source": source,
                    "reason": "profile not found; run start_batch_build_profiles_job first and add only successful papers",
                })
                continue
            normalized.append(clean_ref)

        result = batch_add_papers_to_project(name, normalized)
        if skipped_unprofiled:
            result["skipped"] = [*skipped_unprofiled, *result.get("skipped", [])]
            summary = result.setdefault("summary", {})
            summary["input_count"] = len(papers or [])
            summary["skipped_count"] = len(result["skipped"])
        log_invocation("batch_add_to_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("batch_add_to_project_tool", arguments, error=str(e))
        raise

def clear_project_tool(name: str) -> dict:
    """
    Remove all papers from an existing project manifest.

    Args:
        name: Project name. Will be slugified to match project manifests.

    Returns:
        Structured result with removed_count and updated manifest.
    """
    logger.info("Tool invoked: clear_project name=%r", name)
    arguments = {"name": name}
    try:
        result = clear_project(name)
        log_invocation("clear_project_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("clear_project_tool", arguments, error=str(e))
        raise
