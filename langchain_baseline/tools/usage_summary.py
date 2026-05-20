from langchain_core.tools import tool

from langchain_baseline.services import usage_summary_impl


@tool
def usage_summary() -> dict:
    """Show token usage and cost summary for the current LangChain baseline session."""
    return usage_summary_impl()
