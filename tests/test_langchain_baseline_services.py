from langchain_baseline.services import _compact_search_result


def test_compact_search_result_truncates_abstract():
    result = _compact_search_result({
        "paper_id": "1234.56789",
        "source": "arxiv",
        "title": "A Test Paper",
        "abstract": "x" * 1000,
    })

    assert result["paper_id"] == "1234.56789"
    assert result["source"] == "arxiv"
    assert result["title"] == "A Test Paper"
    assert len(result["abstract"]) < 750
