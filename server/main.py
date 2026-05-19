from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from utils.logger import init_logging, get_logger
from tools.search_papers import search_papers_tool
from tools.batch_ingest_papers import batch_ingest_papers_tool
from tools.detect_gaps import detect_gaps_tool
from tools.manage_project import (
    create_project_tool,
    batch_add_to_project_tool,
    clear_project_tool,
)
from tools.suggest_experiments import suggest_experiments_tool
from tools.validate_gap import validate_gap_tool
from tools.usage_summary import usage_summary_tool
from tools.generate_bibliography import generate_bibliography_tool
from tools.generate_project_report import generate_project_report_tool
from tools.jobs import (
    cancel_job_tool,
    get_job_result_tool,
    get_job_status_tool,
    start_batch_build_profiles_job_tool,
    start_batch_validate_gaps_job_tool,
)

init_logging()
logger = get_logger(__name__)

mcp = FastMCP("research-agent")

mcp.tool()(search_papers_tool)
mcp.tool()(batch_ingest_papers_tool)
mcp.tool()(detect_gaps_tool)
mcp.tool()(create_project_tool)
mcp.tool()(batch_add_to_project_tool)
mcp.tool()(clear_project_tool)
mcp.tool()(suggest_experiments_tool)
mcp.tool()(validate_gap_tool)
mcp.tool()(usage_summary_tool)
mcp.tool()(generate_bibliography_tool)
mcp.tool()(generate_project_report_tool)
mcp.tool(name="start_batch_build_profiles_job")(start_batch_build_profiles_job_tool)
mcp.tool(name="start_batch_validate_gaps_job")(start_batch_validate_gaps_job_tool)
mcp.tool()(get_job_status_tool)
mcp.tool()(get_job_result_tool)
mcp.tool()(cancel_job_tool)

from prompts import register_prompts
register_prompts(mcp)

if __name__ == "__main__":
    logger.info("Starting research-agent MCP server")
    mcp.run()
