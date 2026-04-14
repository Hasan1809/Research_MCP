import json
import os
from services.analysis.synthesis import synthesize_insights
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)

_INSIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "insights")


def synthesize_papers_tool(papers: list[dict]) -> dict:
    logger.info("Tool invoked: synthesize_papers count=%d", len(papers))
    arguments = {"papers": papers}

    insights = []
    for ref in papers:
        paper_id = ref.get("paper_id")
        source = ref.get("source")
        path = os.path.join(_INSIGHTS_DIR, source, f"{paper_id}.json")
        if not os.path.exists(path):
            error = f"Insights not found for paper_id={paper_id!r} source={source!r}. Run extract_paper_insights_tool first."
            log_invocation("synthesize_papers_tool", arguments, error=error)
            raise FileNotFoundError(error)
        logger.info("Loading insights: %s", path)
        with open(path, encoding="utf-8") as f:
            insights.append(json.load(f))

    try:
        result = synthesize_insights(insights)
        log_invocation("synthesize_papers_tool", arguments, output=result)
        return result
    except Exception as e:
        log_invocation("synthesize_papers_tool", arguments, error=str(e))
        raise
