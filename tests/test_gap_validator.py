import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.analysis import gap_validator
from tools import validate_gap as validate_gap_tool_module


def _paper(
    paper_id: str,
    title: str,
    abstract: str,
    *,
    source: str = "semantic_scholar",
    year: str = "2025",
) -> dict:
    return {
        "paper_id": paper_id,
        "source": source,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": ["A. Researcher"],
        "doi": f"10.0000/{paper_id}",
        "url": f"https://example.test/{paper_id}",
        "pdf_url": f"https://example.test/{paper_id}.pdf",
        "semantic_scholar_id": paper_id if source == "semantic_scholar" else "",
        "arxiv_id": paper_id if source == "arxiv" else "",
    }


def test_query_generation_from_gap():
    queries = gap_validator.generate_gap_validation_queries(
        "Lack of real-world evaluation of LLM agent security defenses"
    )

    assert 3 <= len(queries) <= 5
    assert any("llm" in query.lower() for query in queries)
    assert any("evaluation" in query.lower() for query in queries)


def test_classification_categories():
    gap = "Lack of real-world evaluation of LLM agent security defenses"

    direct = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "direct",
            "Real-world evaluation of LLM agent security defenses",
            "We evaluate deployed tool-use agent defenses in production settings.",
        ),
    )
    partial = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "partial",
            "Benchmark for LLM agent security defenses",
            "We introduce a simulated benchmark and taxonomy for tool-use security.",
        ),
    )
    related = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "related",
            "Security risks in LLM agents",
            "We discuss vulnerabilities in language model agents and tool use.",
        ),
    )
    irrelevant = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "irrelevant",
            "Graph neural networks for molecule generation",
            "We study molecular property prediction.",
        ),
    )

    assert direct["classification"] == "directly_addresses_gap"
    assert partial["classification"] == "partially_addresses_gap"
    assert related["classification"] == "related_but_not_addressing"
    assert irrelevant["classification"] == "irrelevant"


def test_validation_with_mocked_search_results_and_artifact(tmp_path, monkeypatch):
    gap = "Lack of real-world evaluation of LLM agent security defenses"
    search_results = [
        _paper(
            "external-direct-1",
            "Real-world evaluation of LLM agent security defenses",
            "We evaluate deployed tool-use agent security defenses in production.",
        ),
        _paper(
            "external-direct-2",
            "Production evaluation for tool-use agent security",
            "This empirical study evaluates LLM agent defenses in real-world deployments.",
            source="arxiv",
        ),
    ]

    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: search_results)

    result = gap_validator.validate_gap(gap, max_results=5)

    assert result["status"] == "already_addressed"
    assert result["confidence"] == "high"
    assert result["results_found"] == 2
    assert result["relevant_results"] == 2
    assert result["artifact_path"]
    assert os.path.exists(result["artifact_path"])

    saved = json.loads(open(result["artifact_path"], encoding="utf-8").read())
    assert saved["gap"] == gap
    assert saved["validation_papers"][0]["classification"] == "directly_addresses_gap"


def test_project_aware_marking_of_existing_and_external_papers(tmp_path, monkeypatch):
    gap = "Lack of real-world evaluation of LLM agent security defenses"
    project_papers = [{"paper_id": "project-paper", "source": "semantic_scholar"}]
    search_results = [
        _paper(
            "project-paper",
            "Real-world evaluation of LLM agent security defenses",
            "We evaluate deployed defenses.",
        ),
        _paper(
            "external-paper",
            "Benchmark for LLM agent security defenses",
            "We introduce a benchmark for tool-use security.",
        ),
    ]

    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "get_project_papers", lambda project: project_papers)
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: search_results)

    result = gap_validator.validate_gap(gap, project="agent-security", max_results=5)
    by_id = {paper["paper_id"]: paper for paper in result["validation_papers"]}

    assert by_id["project-paper"]["already_in_project"] is True
    assert by_id["project-paper"]["external_to_project"] is False
    assert by_id["external-paper"]["already_in_project"] is False
    assert by_id["external-paper"]["external_to_project"] is True


def test_output_schema_contains_required_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: [])

    result = gap_validator.validate_gap(
        "Lack of longitudinal production deployment studies of LLM agent defenses",
        max_results=2,
    )

    expected = {
        "gap",
        "project",
        "mode",
        "search_queries",
        "results_found",
        "relevant_results",
        "status",
        "confidence",
        "decision_reason",
        "refined_gap",
        "validation_papers",
        "recommended_next_step",
        "artifact_path",
    }
    assert expected <= set(result.keys())
    assert result["status"] == "insufficient_evidence"


def test_validate_gap_tool_wrapper_is_thin_and_returns_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: [])
    monkeypatch.setattr(
        validate_gap_tool_module,
        "log_invocation",
        lambda *_args, **_kwargs: None,
    )

    result = validate_gap_tool_module.validate_gap_tool(
        gap="Lack of longitudinal production deployment studies of LLM agent defenses",
        max_results=2,
    )

    assert result["mode"] == "metadata_only"
    assert os.path.exists(result["artifact_path"])
