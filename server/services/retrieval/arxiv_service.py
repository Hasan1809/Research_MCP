import httpx
import xml.etree.ElementTree as ET


def fetch_papers(query: str, limit: int) -> list[dict]:
    response = httpx.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": query, "max_results": limit},
    )
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    return [
        {
            "title": entry.findtext("atom:title", namespaces=ns).strip(),
            "abstract": entry.findtext("atom:summary", namespaces=ns).strip(),
        }
        for entry in root.findall("atom:entry", namespaces=ns)
    ]
