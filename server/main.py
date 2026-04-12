from mcp.server.fastmcp import FastMCP
from tools.search_papers import search_papers_tool

mcp = FastMCP("research-agent")

mcp.tool()(search_papers_tool)

if __name__ == "__main__":
    mcp.run()
