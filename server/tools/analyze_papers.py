from services.analysis.synthesis import analyze_papers
from utils.logger import get_logger, log_invocation

logger = get_logger(__name__)


def analyze_papers_tool(papers: list[dict]) -> dict:
    logger.info("Tool invoked: analyze_papers count=%d", len(papers))
    try:
        result = analyze_papers(papers)
        logger.info("Analysis complete: themes=%s", result["themes"])
        log_invocation("analyze_papers_tool", {"papers": papers}, output=result)
        return result
    except Exception as e:
        log_invocation("analyze_papers_tool", {"papers": papers}, error=str(e))
        raise
