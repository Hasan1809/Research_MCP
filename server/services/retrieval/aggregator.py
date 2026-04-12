from services.retrieval import arxiv_service, semantic_scholar_service


def fetch_papers(query: str, limit: int) -> list[dict]:
    return (
        arxiv_service.fetch_papers(query, limit)
        + semantic_scholar_service.fetch_papers(query, limit)
    )
