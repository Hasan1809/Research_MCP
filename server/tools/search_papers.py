from services.retrieval.arxiv_service import fetch_papers


def search_papers_tool(query: str, limit: int) -> list[dict]:
    return fetch_papers(query, limit)


