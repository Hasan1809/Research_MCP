"""MCP tool for batch-validating project research gaps."""
from services.analysis.gap_validator import batch_validate_gaps
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def batch_validate_gaps_tool(
    project: str,
    max_results_per_gap: int = 10,
    mode: str = "metadata_only",
    max_workers: int = 2,
) -> dict:
    """
    Validate all candidate research and methodological gaps for a project.

    This tool performs targeted follow-up academic search for each gap using
    existing arXiv and Semantic Scholar retrieval services. It classifies each
    gap as confirmed, partially addressed, already addressed, too broad, needing
    refinement, or insufficiently evidenced. It saves individual validation
    artifacts and a compact batch summary that can be used by
    suggest_experiments_tool.

    Run this after detect_gaps_tool(project=...) and before
    suggest_experiments_tool(project=..., compact=True) when you want experiment
    suggestions to use validated/refined gaps.
    """
    arguments = {
        "project": project,
        "max_results_per_gap": max_results_per_gap,
        "mode": mode,
        "max_workers": max_workers,
    }
    logger.info(
        "Tool invoked: batch_validate_gaps project=%r max_results_per_gap=%d mode=%s max_workers=%d",
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
        log_invocation("batch_validate_gaps_tool", arguments, error=str(e))
        raise

    log_invocation("batch_validate_gaps_tool", arguments, output={
        "project": result.get("project"),
        "gap_count": result.get("gap_count"),
        "validated_count": result.get("validated_count"),
        "failed_count": result.get("failed_count"),
        "status_counts": result.get("status_counts"),
        "batch_artifact_path": result.get("batch_artifact_path"),
    })
    return result
