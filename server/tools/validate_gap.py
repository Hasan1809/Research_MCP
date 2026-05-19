"""MCP tool for validating candidate research gaps."""
from services.analysis.gap_validator import validate_gap
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def validate_gap_tool(
    gap: str,
    project: str = None,
    max_results: int = 10,
    mode: str = "metadata_only",
) -> dict:
    """
    Validate a candidate research gap by performing targeted follow-up search across academic sources.

    This tool checks whether a gap detected from a project paper set is already
    addressed by wider literature. It returns a validation status, supporting
    papers, and an optional refined gap. It does not prove a gap absolutely; it
    classifies the available evidence.

    Use after detect_gaps_tool when a candidate gap needs field-level checking.
    Prefer passing project when the gap came from a saved project so validation
    papers can be marked as already_in_project or external_to_project.

    The default mode is metadata_only: the tool searches existing arXiv and
    Semantic Scholar services and classifies title/abstract metadata without
    ingesting PDFs.
    """
    arguments = {
        "gap": gap,
        "project": project,
        "max_results": max_results,
        "mode": mode,
    }
    logger.info(
        "Tool invoked: validate_gap project=%r max_results=%d mode=%s",
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
        log_invocation("validate_gap_tool", arguments, error=str(e))
        raise

    log_invocation("validate_gap_tool", arguments, output={
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "results_found": result.get("results_found"),
        "relevant_results": result.get("relevant_results"),
        "artifact_path": result.get("artifact_path"),
    })
    return result
