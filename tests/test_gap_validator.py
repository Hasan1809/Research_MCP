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
    assert {"score", "matched_facets", "missing_facets", "decision_reason"} <= set(direct)


def test_general_cybersecurity_does_not_directly_address_llm_agent_gap():
    gap = "Human-centered evaluation of LLM agent security for real-world users"

    result = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "kernel",
            "SyzScope: Revealing High-Risk Security Impacts of Fuzzer-Exposed Bugs in Linux kernel",
            "We study kernel fuzzing and vulnerability impact analysis for operating systems.",
        ),
    )

    assert result["classification"] in {"related_but_not_addressing", "irrelevant"}
    assert result["classification"] != "directly_addresses_gap"


def test_generic_real_world_adjacent_domains_do_not_close_llm_agent_deployment_gap():
    gap = "Evaluation of LLM-integrated app systems in real-world, dynamic environments"
    adjacent = [
        _paper(
            "driver",
            "A Dynamic-System-Based Approach to Modeling Driver Movements Across Managed Lane Interfaces",
            "We evaluate real-world driver behavior in dynamic transportation environments.",
        ),
        _paper(
            "robot",
            "Robot Policy Evaluation for Sim-to-Real Transfer",
            "We benchmark robotics policy evaluation under sim-to-real deployment constraints.",
        ),
        _paper(
            "hci",
            "Reporting and Reviewing LLM-Integrated Systems in HCI",
            "We discuss reporting practices for HCI systems, without security attacks or vulnerabilities.",
        ),
    ]

    classifications = [
        gap_validator.classify_validation_paper(
            gap,
            paper,
            project_context="llm agent security",
        )["classification"]
        for paper in adjacent
    ]

    assert "directly_addresses_gap" not in classifications


def test_llm_agent_benchmark_without_users_partially_addresses_human_gap():
    gap = "Human-centered evaluation of LLM agent security for real-world users"

    result = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "benchmark",
            "Benchmarking LLM agent security defenses",
            "We evaluate tool-using LLM agents with a simulated security benchmark but do not run a user study.",
        ),
    )

    assert result["classification"] == "partially_addresses_gap"
    assert any("real-world" in facet or "users" in facet for facet in result["missing_facets"])


def test_llm_agent_user_study_directly_addresses_human_gap():
    gap = "Human-centered evaluation of LLM agent security for real-world users"

    result = gap_validator.classify_validation_paper(
        gap,
        _paper(
            "user-study",
            "Human-centered real-world evaluation of LLM agent security for users",
            "We evaluate language model agents in realistic tool-use security tasks with non-technical users and participants.",
        ),
    )

    assert result["classification"] == "directly_addresses_gap"
    assert result["score"] >= 0.72


def test_two_strong_matches_close_gap_with_medium_confidence(tmp_path, monkeypatch):
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
    monkeypatch.setattr(gap_validator, "get_project_papers", lambda project: [])
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: search_results)

    result = gap_validator.validate_gap(gap, max_results=5)

    assert result["status"] == "already_addressed"
    assert result["confidence"] == "medium"
    assert result["results_found"] == 2
    assert result["relevant_results"] == 2
    assert result["artifact_path"]
    assert os.path.exists(result["artifact_path"])

    saved = json.loads(open(result["artifact_path"], encoding="utf-8").read())
    assert saved["gap"] == gap
    assert saved["validation_papers"][0]["classification"] == "directly_addresses_gap"
    assert {"score", "matched_facets", "missing_facets", "decision_reason"} <= set(
        saved["validation_papers"][0]
    )


def test_three_strong_external_matches_can_close_gap(tmp_path, monkeypatch):
    gap = "Lack of real-world evaluation of LLM agent security defenses"
    search_results = [
        _paper(
            "external-direct-1",
            "Real-world evaluation of LLM agent security defenses",
            "We evaluate deployed tool-use LLM agent security defenses in production.",
        ),
        _paper(
            "external-direct-2",
            "Production evaluation for LLM agent security defenses",
            "This empirical study evaluates language model agent defenses in real-world deployments.",
            source="arxiv",
        ),
        _paper(
            "external-direct-3",
            "Field study of LLM agent security defenses for tool use",
            "We study real-world deployed LLM agents and evaluate security defenses for tool-use attacks.",
        ),
    ]

    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "get_project_papers", lambda project: [])
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: search_results)

    result = gap_validator.validate_gap(gap, max_results=5)

    assert result["status"] == "already_addressed"
    assert result["confidence"] == "high"


def test_two_weak_related_matches_do_not_produce_already_addressed(tmp_path, monkeypatch):
    gap = "Human-centered evaluation of LLM agent security for real-world users"
    search_results = [
        _paper(
            "kernel",
            "Kernel fuzzing for security vulnerabilities",
            "We evaluate Linux kernel vulnerability detection.",
        ),
        _paper(
            "vehicles",
            "Connected vehicle security vulnerability detection",
            "We study attacks and defenses in VANET and IoT systems.",
        ),
    ]

    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: search_results)

    result = gap_validator.validate_gap(gap, max_results=5)

    assert result["status"] != "already_addressed"
    assert result["confidence"] != "high"


def test_explainability_refined_gap_is_actionable(tmp_path, monkeypatch):
    gap = "Application of explainability techniques to LLM agent security"
    search_results = [
        _paper(
            "generic-xai",
            "Explainability methods for machine learning systems",
            "We survey SHAP and LIME for general model interpretability.",
        ),
        _paper(
            "agent-security",
            "Security risks in LLM agents",
            "We discuss prompt injection and unsafe tool choices in language model agents.",
        ),
        _paper(
            "agent-benchmark",
            "Benchmarking LLM agent security",
            "We evaluate prompt injection attacks against tool-using LLM agents.",
        ),
    ]

    monkeypatch.setattr(gap_validator, "_VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(gap_validator, "get_project_papers", lambda project: [])
    monkeypatch.setattr(gap_validator, "fetch_papers", lambda query, limit: search_results)

    result = gap_validator.validate_gap(gap, project="llm-agent-security", max_results=5)

    assert result["status"] in {"partially_addressed", "needs_refinement", "confirmed_candidate_gap"}
    assert "with emphasis on LLM agents, security" not in result["refined_gap"]
    if result["refined_gap"]:
        assert "unsafe tool choices" in result["refined_gap"] or "prompt-injection" in result["refined_gap"]


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
