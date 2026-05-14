from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from utils.logger import init_logging, get_logger
from tools.search_papers import search_papers_tool
from tools.ingest_paper import ingest_paper_tool
from tools.batch_ingest_papers import batch_ingest_papers_tool
from tools.index_paper import index_paper_tool
from tools.retrieve_chunks import retrieve_chunks_tool
from tools.build_paper_profile import build_paper_profile_tool
from tools.batch_build_profiles import batch_build_profiles_tool
from tools.detect_gaps import detect_gaps_tool
from tools.manage_project import (
    create_project_tool,
    add_to_project_tool,
    batch_add_to_project_tool,
    list_projects_tool,
)
from tools.suggest_experiments import suggest_experiments_tool
from tools.usage_summary import usage_summary_tool
from tools.generate_bibliography import generate_bibliography_tool

init_logging()
logger = get_logger(__name__)

mcp = FastMCP("research-agent")

mcp.tool()(search_papers_tool)
mcp.tool()(ingest_paper_tool)
mcp.tool()(index_paper_tool)
mcp.tool()(retrieve_chunks_tool)
mcp.tool()(build_paper_profile_tool)
mcp.tool()(detect_gaps_tool)
mcp.tool()(create_project_tool)
mcp.tool()(add_to_project_tool)
mcp.tool()(batch_add_to_project_tool)
mcp.tool()(list_projects_tool)
mcp.tool()(suggest_experiments_tool)
mcp.tool()(usage_summary_tool)
mcp.tool()(batch_ingest_papers_tool)
mcp.tool()(batch_build_profiles_tool)
mcp.tool()(generate_bibliography_tool)

from prompts import register_prompts
register_prompts(mcp)

if __name__ == "__main__":
    logger.info("Starting research-agent MCP server")
    mcp.run()
