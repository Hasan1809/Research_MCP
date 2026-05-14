import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services import citations


def _metadata(title="A {Robust} MCP_Study & Evaluation"):
    return {
        "paper_id": "2401.12345",
        "source": "arxiv",
        "title": title,
        "authors": ["Alice Smith", "Bob Jones"],
        "year": "2024",
        "venue": "arXiv",
        "doi": "10.1000/example",
        "arxiv_id": "2401.12345",
        "semantic_scholar_id": "",
        "url": "https://arxiv.org/abs/2401.12345",
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "source_database": "arxiv",
    }


def test_markdown_bibliography_from_mock_metadata(monkeypatch):
    monkeypatch.setattr(citations, "load_paper_metadata", lambda source, paper_id: _metadata())

    result = citations.generate_bibliography(
        papers=[{"paper_id": "2401.12345", "source": "arxiv"}],
        format="markdown",
        save=False,
    )

    assert result["skipped"] == []
    assert "1. Alice Smith, Bob Jones (2024)." in result["bibliography"]
    assert "A {Robust} MCP_Study & Evaluation" in result["bibliography"]


def test_missing_metadata_is_skipped(monkeypatch):
    monkeypatch.setattr(citations, "load_paper_metadata", lambda source, paper_id: None)
    monkeypatch.setattr(
        citations,
        "normalize_paper_metadata",
        lambda source, paper_id: {"paper_id": paper_id, "source": source, "title": ""},
    )

    result = citations.generate_bibliography(
        papers=[{"paper_id": "missing", "source": "arxiv"}],
        format="bibtex",
        save=False,
    )

    assert result["bibliography"] == ""
    assert result["skipped"][0]["paper_id"] == "missing"
    assert result["skipped"][0]["reason"] == "missing metadata"


def test_bibtex_escaping(monkeypatch):
    monkeypatch.setattr(citations, "load_paper_metadata", lambda source, paper_id: _metadata())

    result = citations.generate_bibliography(
        papers=[{"paper_id": "2401.12345", "source": "arxiv"}],
        format="bibtex",
        save=False,
    )

    assert "\\{Robust\\}" in result["bibliography"]
    assert "MCP\\_Study" in result["bibliography"]
    assert "\\&" in result["bibliography"]


def test_project_based_bibliography(monkeypatch):
    monkeypatch.setattr(
        citations,
        "get_project_papers",
        lambda project_name: [{"paper_id": "2401.12345", "source": "arxiv"}],
    )
    monkeypatch.setattr(citations, "load_paper_metadata", lambda source, paper_id: _metadata("Project Paper"))

    result = citations.generate_bibliography(
        project_name="mcp-project",
        format="markdown",
        save=False,
    )

    assert len(result["included"]) == 1
    assert "Project Paper" in result["bibliography"]
