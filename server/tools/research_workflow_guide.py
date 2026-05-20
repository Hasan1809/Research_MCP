"""Deterministic workflow guide for research-agent orchestration."""

from config import WORKFLOW_MAX_PAPERS
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def _project_name(topic: str, project: str | None) -> str:
    return (project or topic or "research-project").strip()


def get_research_workflow_guide_tool(
    topic: str,
    project: str | None = None,
    num_papers: int = 8,
) -> dict:
    """
    Call this first for normal user requests like "find gaps and suggest experiments for <topic>".

    Returns the canonical research-agent workflow, tool order, guardrails, and
    concise final-answer contract. This tool performs no analysis and calls no LLM;
    it exists so the orchestrating model does not need the user to know the
    internal MCP workflow.
    """
    requested = max(2, min(int(num_papers or 8), WORKFLOW_MAX_PAPERS))
    project_name = _project_name(topic, project)
    result = {
        "topic": topic,
        "project": project_name,
        "paper_selection_limit": requested,
        "normal_max_papers": WORKFLOW_MAX_PAPERS,
        "purpose": (
            "Use this workflow for user requests to find research gaps, validate gaps, "
            "suggest experiments, or create a final research-agent report."
        ),
        "tool_order": [
            {
                "step": 1,
                "tool": "search_papers_tool",
                "instruction": f"Search for the topic and select at most {requested} directly relevant papers.",
            },
            {
                "step": 2,
                "tool": "create_project_tool",
                "instruction": f"Create project {project_name!r} with overwrite=True for a clean run.",
            },
            {
                "step": 3,
                "tool": "batch_ingest_papers_tool",
                "instruction": "Ingest only selected papers. Continue with successful ingestions if some fail.",
            },
            {
                "step": 4,
                "tool": "start_batch_build_profiles_job",
                "instruction": "Build profiles only for successfully ingested papers; poll with get_job_status_tool.",
            },
            {
                "step": 5,
                "tool": "batch_add_to_project_tool",
                "instruction": "Add only successfully profiled papers. The tool skips unprofiled papers.",
            },
            {
                "step": 6,
                "tool": "get_workflow_status_tool",
                "instruction": "Verify profiled_count equals paper_count before detecting gaps.",
            },
            {
                "step": 7,
                "tool": "detect_gaps_tool",
                "instruction": "Detect research and methodological gaps for the project.",
            },
            {
                "step": 8,
                "tool": "start_batch_validate_gaps_job",
                "instruction": "Validate all candidate gaps; poll with get_job_status_tool and fetch result.",
            },
            {
                "step": 9,
                "tool": "suggest_experiments_tool",
                "instruction": "Use compact=True and generate experiments only from included validated gaps.",
            },
            {
                "step": 10,
                "tool": "generate_bibliography_tool",
                "instruction": "Generate BibTeX bibliography for the project.",
            },
            {
                "step": 11,
                "tool": "generate_project_report_tool",
                "instruction": "Generate the deterministic Markdown report.",
            },
            {
                "step": 12,
                "tool": "get_workflow_status_tool",
                "instruction": "Use final status counts and artifact paths in the response.",
            },
        ],
        "guardrails": [
            "Do not create separate Markdown/SVG/index files outside generate_project_report_tool.",
            "Do not add budget, team size, timelines, compute estimates, or implementation plans unless the user asks.",
            "Search results are not a valid final answer.",
            "Do not stop after gap detection or validation; continue through experiments, bibliography, and report.",
            "Only produce the final answer after generate_project_report_tool and the final get_workflow_status_tool call complete.",
            "Do not invent experiments if validation says all gaps are already addressed.",
            "Keep the final answer short and based on tool outputs, not new analysis prose.",
        ],
        "final_answer_contract": {
            "max_paragraphs": 2,
            "required_fields": [
                "project",
                "paper_count",
                "gap_count",
                "included_validated_gap_count",
                "excluded_validated_gap_count",
                "experiment_count",
                "bibliography_path",
                "report_path",
                "warnings",
            ],
            "style": "compact status summary, no extra documents, no long generated report in chat",
        },
    }
    log_invocation(
        "get_research_workflow_guide_tool",
        {"topic": topic, "project": project, "num_papers": num_papers},
        output={
            "project": project_name,
            "paper_selection_limit": requested,
            "step_count": len(result["tool_order"]),
        },
    )
    logger.info("Tool invoked: get_research_workflow_guide topic=%r project=%r", topic, project_name)
    return result
