from utils.usage_tracker import get_usage_summary
from utils.logger import get_logger

logger = get_logger(__name__)


def usage_summary_tool() -> dict:
    """
    Show token usage and cost summary for all IONOS LLM calls.

    Returns total tokens, cost breakdown by tool, model used,
    and average latency. Useful for monitoring costs and
    optimizing the pipeline.
    """
    logger.info("Tool invoked: usage_summary")
    return get_usage_summary()
