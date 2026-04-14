import json
import os
from services.analysis.synthesis import synthesize_insights
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_INSIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "insights")
_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "profiles")


def _profile_to_insights(profile: dict) -> dict:
    """Map paper profile fields to the insight schema used by synthesize_insights."""
    return {
        "paper_id": profile.get("paper_id", ""),
        "source": profile.get("source", ""),
        "methods": (
            [profile["methods_or_approach"]] if profile.get("methods_or_approach") else []
        ),
        "results": (
            [profile["key_findings"]] if profile.get("key_findings") else []
        ),
        "datasets":    profile.get("datasets", []),
        "limitations": profile.get("limitations", []),
        "future_work": profile.get("future_work", []),
    }


def synthesize_papers_tool(papers: list[dict]) -> dict:
    logger.info("Tool invoked: synthesize_papers count=%d", len(papers))
    arguments = {"papers": papers}

    insights = []
    for ref in papers:
        paper_id = ref.get("paper_id")
        source = ref.get("source")
        profile_path = os.path.join(_PROFILES_DIR, source, f"{paper_id}.json")
        insights_path = os.path.join(_INSIGHTS_DIR, source, f"{paper_id}.json")

        if os.path.exists(profile_path):
            logger.info("Loading profile for %r (preferred over insights)", paper_id)
            with open(profile_path, encoding="utf-8") as f:
                insights.append(_profile_to_insights(json.load(f)))
        elif os.path.exists(insights_path):
            logger.info("Loading insights for %r (no profile found)", paper_id)
            with open(insights_path, encoding="utf-8") as f:
                insights.append(json.load(f))
        else:
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
