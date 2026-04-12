from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from utils.logger import init_logging, get_logger
from tools.search_papers import search_papers_tool
from tools.analyze_papers import analyze_papers_tool
from tools.ingest_paper import ingest_paper_tool

init_logging()
logger = get_logger(__name__)

mcp = FastMCP("research-agent")

mcp.tool()(search_papers_tool)
mcp.tool()(analyze_papers_tool)
mcp.tool()(ingest_paper_tool)

if __name__ == "__main__":
    logger.info("Starting research-agent MCP server")
    mcp.run()
