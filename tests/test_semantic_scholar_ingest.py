import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from tools import ingest_paper


def test_semantic_scholar_ingest_resolves_pdf_and_caches(monkeypatch):
    saved = {}

    monkeypatch.setattr(ingest_paper, "load_cached", lambda source, paper_id: None)
    monkeypatch.setattr(ingest_paper, "log_invocation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingest_paper,
        "resolve_pdf_url",
        lambda paper_id: (
            "https://example.test/paper.pdf",
            {"paper_id": paper_id, "pdf_url": "https://example.test/paper.pdf"},
        ),
    )
    monkeypatch.setattr(
        ingest_paper,
        "download_and_extract_text",
        lambda pdf_url: "A Semantic Scholar Paper\n\nAbstract\n\nThis is the abstract.",
    )

    def fake_save_cached(source, paper_id, data):
        saved["source"] = source
        saved["paper_id"] = paper_id
        saved["data"] = data

    monkeypatch.setattr(ingest_paper, "save_cached", fake_save_cached)

    result = ingest_paper.ingest_paper_tool(
        "649def34f8be52c8b66281af98ae884c09aef38b",
        "semantic_scholar",
    )

    assert result["source"] == "semantic_scholar"
    assert result["pdf_url"] == "https://example.test/paper.pdf"
    assert result["semantic_scholar"]["paper_id"] == "649def34f8be52c8b66281af98ae884c09aef38b"
    assert saved["source"] == "semantic_scholar"
    assert saved["data"]["chunk_count"] > 0
