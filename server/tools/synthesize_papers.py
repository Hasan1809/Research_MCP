from services.analysis.synthesis import synthesize_insights
from services.paper_repository import load_insights, load_profile
from services.project_manager import get_project_papers
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def _profile_to_insights(profile: dict) -> dict:
    return {
        "paper_id": profile.get("paper_id", ""),
        "source": profile.get("source", ""),
        "methods": [profile["methods_or_approach"]] if profile.get("methods_or_approach") else [],
        "results": [profile["key_findings"]] if profile.get("key_findings") else [],
        "datasets": profile.get("datasets", []),
        "limitations": profile.get("limitations", []),
        "future_work": profile.get("future_work", []),
    }


def synthesize_papers_tool(papers: list[dict] = None, project: str = None) -> dict:
    if project:
        papers = get_project_papers(project)
        logger.info("Tool invoked: synthesize_papers project=%r count=%d", project, len(papers))
    elif papers:
        logger.info("Tool invoked: synthesize_papers count=%d", len(papers))
    else:
        raise ValueError("synthesize_papers_tool requires either 'papers' or 'project'.")

    arguments = {"papers": papers, "project": project}

    insights = []
    for ref in papers:
        paper_id = ref.get("paper_id")
        source = ref.get("source")

        profile = load_profile(source, paper_id)
        if profile is not None:
            logger.info("Loading profile for %r (preferred over insights)", paper_id)
            insights.append(_profile_to_insights(profile))
            continue

        insight = load_insights(source, paper_id)
        if insight is not None:
            logger.info("Loading insights for %r (no profile found)", paper_id)
            insights.append(insight)
            continue

        error = (
            f"No profile or insights found for paper_id={paper_id!r} source={source!r}. "
            "Run build_paper_profile_tool or extract_paper_insights_tool first."
        )
        log_invocation("synthesize_papers_tool", arguments, error=error)
        raise FileNotFoundError(error)

    try:
        result = synthesize_insights(insights)
        log_invocation("synthesize_papers_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("synthesize_papers_tool", arguments, error=str(e))
        raise
