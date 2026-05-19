import json
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.analysis import experiment_suggester
from services.analysis import gap_validator
from services import project_manager
from tools import suggest_experiments as suggest_experiments_tool_module


def _papers(count: int) -> list[dict]:
    return [
        {"paper_id": f"paper-{index}", "source": "test"}
        for index in range(count)
    ]


def _gap_analysis(paper_ids: list[str]) -> dict:
    return {
        "research_gaps": [
            {
                "gap": "Real-world tool-use robustness is under-evaluated",
                "evidence": "Existing evaluations use narrow scenarios.",
                "relevant_papers": paper_ids[:3],
            }
        ],
        "methodological_gaps": [
            {
                "gap": "No cross-benchmark stress test",
                "current_approaches": ["single benchmark"],
                "missing_approach": "multi-benchmark transfer evaluation",
                "relevant_papers": paper_ids[2:5],
            }
        ],
        "contradictions": [
            {
                "finding_a": "Guardrails improve safety",
                "finding_b": "Guardrails can be bypassed",
                "paper_a": paper_ids[0],
                "paper_b": paper_ids[1],
                "nature": "different threat models",
            }
        ],
        "connections": [
            {
                "insight": "Benchmark and defense papers can be combined",
                "papers": paper_ids[:2],
                "potential": "stronger evaluation",
            }
        ],
        "field_summary": "The field lacks broad dynamic evaluations.",
    }


def _write_gap_artifact(tmp_path, papers, analysis):
    path = tmp_path / "gap_analysis_20260514_test.json"
    path.write_text(
        json.dumps({"papers": papers, "analysis": analysis}),
        encoding="utf-8",
    )
    return path


class FakeLLM:
    calls = []
    response = {
        "experiments": [
            {
                "title": "Cross-benchmark stress test",
                "addresses_gap": "No cross-benchmark stress test",
                "hypothesis": "Transfer attacks expose missed failures.",
                "method": "Run the same attacks across benchmark suites.",
                "baselines": ["single benchmark evaluation"],
                "datasets": ["existing benchmark tasks"],
                "expected_outcome": "Finds non-transferable defenses.",
                "feasibility": "high",
                "builds_on": ["paper-0", "outside-paper"],
            }
        ]
    }

    def call(self, **kwargs):
        self.__class__.calls.append(kwargs)
        return self.response, json.dumps(self.response)


def test_uses_existing_gap_analysis_cache_without_detecting_or_loading_profiles(tmp_path, monkeypatch):
    papers = _papers(3)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    cached_path = _write_gap_artifact(tmp_path, papers, analysis)
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_profile_or_insights",
        lambda *_args: (_ for _ in ()).throw(AssertionError("profiles should not load")),
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_profile",
        lambda *_args: (_ for _ in ()).throw(AssertionError("full profiles should not load")),
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(papers, compact=True)

    assert result["gap_source"] == "cache"
    assert result["gap_analysis_path"] == str(cached_path)
    assert result["paper_count"] == 3
    assert result["subset_used"] is False
    assert FakeLLM.calls


def test_tool_uses_existing_gap_analysis_cache(tmp_path, monkeypatch):
    papers = _papers(3)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    cached_path = _write_gap_artifact(tmp_path, papers, analysis)
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )
    monkeypatch.setattr(
        suggest_experiments_tool_module,
        "log_invocation",
        lambda *_args, **_kwargs: None,
    )

    result = suggest_experiments_tool_module.suggest_experiments_tool(
        papers=papers,
        compact=True,
    )

    assert result["gap_source"] == "cache"
    assert result["gap_analysis_path"] == str(cached_path)
    assert result["paper_count"] == 3


def test_project_only_invocation_loads_project_manifest_and_returns_compact_metadata(
    tmp_path,
    monkeypatch,
):
    papers = _papers(3)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    cached_path = _write_gap_artifact(tmp_path, papers, analysis)
    invocation_records = []
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(project_manager, "get_project_papers", lambda project: papers)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )
    monkeypatch.setattr(
        suggest_experiments_tool_module,
        "log_invocation",
        lambda *args, **kwargs: invocation_records.append((args, kwargs)),
    )

    result = suggest_experiments_tool_module.suggest_experiments_tool(
        project="saved-project",
        compact=True,
    )

    assert result["project"] == "saved-project"
    assert result["active_paper_count"] == 3
    assert result["experiment_count"] == 1
    assert result["gap_count"] == 2
    assert result["gap_source"] == "cache"
    assert result["gap_analysis_path"] == str(cached_path)
    assert result["compact"] is True
    assert result["error"] is None

    invocation_args = invocation_records[0][0][1]
    invocation_output = invocation_records[0][1]["output"]
    assert invocation_args == {
        "project": "saved-project",
        "active_paper_count": 3,
        "compact": True,
    }
    assert "papers" not in invocation_args
    assert "papers" not in invocation_output
    assert invocation_output["project"] == "saved-project"
    assert invocation_output["active_paper_count"] == 3


def test_explicit_papers_only_invocation_still_works(tmp_path, monkeypatch):
    papers = _papers(2)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )
    monkeypatch.setattr(
        suggest_experiments_tool_module,
        "log_invocation",
        lambda *_args, **_kwargs: None,
    )

    result = suggest_experiments_tool_module.suggest_experiments_tool(
        papers=papers,
        compact=True,
    )

    assert result["project"] is None
    assert result["active_paper_count"] == 2
    assert result["gap_source"] == "cache"
    assert result["compact"] is True


def test_project_and_papers_invocation_prefers_project_and_ignores_papers(
    tmp_path,
    monkeypatch,
    caplog,
):
    project_papers = _papers(3)
    ignored_papers = [{"paper_id": "ignored-paper", "source": "test"}]
    analysis = _gap_analysis([p["paper_id"] for p in project_papers])
    _write_gap_artifact(tmp_path, project_papers, analysis)
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(project_manager, "get_project_papers", lambda project: project_papers)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )
    monkeypatch.setattr(
        suggest_experiments_tool_module,
        "log_invocation",
        lambda *_args, **_kwargs: None,
    )

    result = suggest_experiments_tool_module.suggest_experiments_tool(
        project="saved-project",
        papers=ignored_papers,
        compact=True,
    )

    assert result["project"] == "saved-project"
    assert result["active_paper_count"] == 3
    assert result["gap_source"] == "cache"
    assert result["experiments"][0]["builds_on"] == ["paper-0"]
    assert "Both project and papers provided; using project manifest and ignoring papers." in caplog.text
    assert "ignored-paper" not in FakeLLM.calls[0]["user"]


def test_suggest_experiments_uses_batch_validation_summary(tmp_path, monkeypatch):
    papers = _papers(2)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    batch_summary = {
        "project": "saved-project",
        "project_papers": papers,
        "batch_artifact_path": str(tmp_path / "batch.json"),
        "validated_gaps": [
            {
                "gap_id": "research_gap_1",
                "gap_type": "research_gap",
                "original_gap": "Original valid gap",
                "status": "partially_addressed",
                "confidence": "medium",
                "use_for_experiments": True,
                "decision_reason": "partly covered",
                "refined_gap": "Refined valid gap",
                "external_evidence_titles": ["External evidence paper"],
            },
            {
                "gap_id": "research_gap_2",
                "gap_type": "research_gap",
                "original_gap": "Already addressed gap",
                "status": "already_addressed",
                "confidence": "high",
                "use_for_experiments": False,
                "decision_reason": "directly addressed",
                "refined_gap": None,
                "external_evidence_titles": ["External addressed paper"],
            },
        ],
        "failed_gaps": [],
    }
    FakeLLM.calls = []
    monkeypatch.setattr(
        FakeLLM,
        "response",
        {
            "experiments": [
                {
                    "title": "Validated experiment",
                    "addresses_gap": "Refined valid gap",
                    "hypothesis": "h",
                    "method": "m",
                    "baselines": [],
                    "datasets": [],
                    "expected_outcome": "o",
                    "feasibility": "high",
                    "builds_on": ["paper-0", "external-paper"],
                }
            ]
        },
    )

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "find_latest_batch_validation_summary",
        lambda project, papers=None: batch_summary,
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(
        papers,
        project="saved-project",
        compact=True,
    )

    user_prompt = FakeLLM.calls[0]["user"]
    assert result["validation_used"] is True
    assert result["batch_validation_path"] == str(tmp_path / "batch.json")
    assert result["included_gap_count"] == 1
    assert result["excluded_gap_count"] == 1
    assert result["refined_gap_count"] == 1
    assert result["excluded_gaps"][0]["status"] == "already_addressed"
    assert "Refined valid gap" in user_prompt
    assert "Already addressed gap" not in user_prompt
    assert "External evidence paper" in user_prompt
    assert result["experiments"][0]["builds_on"] == ["paper-0"]


def test_suggest_experiments_no_validated_gaps_skips_llm(tmp_path, monkeypatch):
    papers = _papers(2)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    batch_summary = {
        "project": "saved-project",
        "project_papers": papers,
        "batch_artifact_path": str(tmp_path / "batch.json"),
        "validated_gaps": [
            {
                "gap_id": "research_gap_1",
                "gap_type": "research_gap",
                "original_gap": "Already addressed gap one",
                "status": "already_addressed",
                "confidence": "high",
                "use_for_experiments": False,
                "decision_reason": "directly addressed",
                "refined_gap": None,
                "external_evidence_titles": ["External addressed paper"],
            },
            {
                "gap_id": "methodological_gap_1",
                "gap_type": "methodological_gap",
                "original_gap": "Already addressed gap two",
                "status": "already_addressed",
                "confidence": "high",
                "use_for_experiments": False,
                "decision_reason": "directly addressed",
                "refined_gap": None,
                "external_evidence_titles": [],
            },
        ],
        "failed_gaps": [],
    }
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "find_latest_batch_validation_summary",
        lambda project, papers=None: batch_summary,
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(
        papers,
        project="saved-project",
        compact=True,
    )

    assert FakeLLM.calls == []
    assert result["status"] == "no_validated_gaps"
    assert result["error"] is None
    assert result["experiments"] == []
    assert result["experiment_count"] == 0
    assert result["validation_used"] is True
    assert result["included_gap_count"] == 0
    assert result["excluded_gap_count"] == 2
    assert len(result["excluded_gaps"]) == 2
    assert result["recommended_next_step"]
    assert "No experiments were generated" in result["recommended_next_step"]
    assert result["save_path"]
    assert os.path.exists(result["save_path"])
    assert result["requested_experiment_count"] == 0
    assert result["raw_experiment_count"] == 0


def test_suggest_experiments_one_included_gap_limits_and_dedupes(tmp_path, monkeypatch):
    papers = _papers(3)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    batch_summary = {
        "project": "saved-project",
        "project_papers": papers,
        "batch_artifact_path": str(tmp_path / "batch.json"),
        "validated_gaps": [
            {
                "gap_id": "methodological_gap_1",
                "gap_type": "methodological_gap",
                "original_gap": "Explainability for LLM agent security",
                "status": "needs_refinement",
                "confidence": "medium",
                "use_for_experiments": True,
                "decision_reason": "needs refinement",
                "refined_gap": "Evaluating whether explainability methods help developers identify unsafe tool choices in LLM agents under prompt-injection attacks.",
                "external_evidence_titles": [],
            }
        ],
        "failed_gaps": [],
    }
    FakeLLM.calls = []
    monkeypatch.setattr(
        FakeLLM,
        "response",
        {
            "experiments": [
                {
                    "title": "Explain unsafe tool choices",
                    "addresses_gap": "Evaluating whether explainability methods help developers identify unsafe tool choices in LLM agents under prompt-injection attacks.",
                    "hypothesis": "Explanations help developers find unsafe tool choices.",
                    "method": "Compare explanation-assisted review against normal review for prompt-injection tool-use traces.",
                    "baselines": [],
                    "datasets": [],
                    "expected_outcome": "Higher detection of unsafe choices.",
                    "feasibility": "high",
                    "builds_on": ["paper-0"],
                },
                {
                    "title": "Explainability for unsafe tool decisions",
                    "addresses_gap": "Evaluating whether explainability methods help developers identify unsafe tool choices in LLM agents under prompt-injection attacks.",
                    "hypothesis": "Explanations help developers identify unsafe tool choices.",
                    "method": "Use SHAP and LIME explanations to compare developer review of prompt-injection tool-use traces.",
                    "baselines": [],
                    "datasets": [],
                    "expected_outcome": "Higher detection of unsafe choices.",
                    "feasibility": "medium",
                    "builds_on": ["paper-0", "paper-1"],
                },
                {
                    "title": "Warning comprehension for tool-use attacks",
                    "addresses_gap": "Evaluating whether explainability methods help developers identify unsafe tool choices in LLM agents under prompt-injection attacks.",
                    "hypothesis": "Decision-transparent warnings improve response quality.",
                    "method": "Run a small developer study comparing warnings with and without security decision traces.",
                    "baselines": [],
                    "datasets": [],
                    "expected_outcome": "Developers handle risky actions better.",
                    "feasibility": "medium",
                    "builds_on": ["paper-2"],
                },
            ]
        },
    )

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "find_latest_batch_validation_summary",
        lambda project, papers=None: batch_summary,
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(
        papers,
        project="saved-project",
        compact=True,
    )

    assert result["included_gap_count"] == 1
    assert result["requested_experiment_count"] == 2
    assert result["raw_experiment_count"] == 3
    assert result["experiment_count"] <= 2
    assert result["deduplicated_experiment_count"] <= 2
    assert "Requested experiment count: 2" in FakeLLM.calls[0]["user"]


def test_suggest_experiments_timeout_keeps_validation_metadata(tmp_path, monkeypatch):
    papers = _papers(2)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    batch_summary = {
        "project": "saved-project",
        "project_papers": papers,
        "batch_artifact_path": str(tmp_path / "batch.json"),
        "validated_gaps": [
            {
                "gap_id": "research_gap_1",
                "gap_type": "research_gap",
                "original_gap": "Original valid gap",
                "status": "confirmed_candidate_gap",
                "confidence": "medium",
                "use_for_experiments": True,
                "decision_reason": "still open",
                "refined_gap": None,
                "external_evidence_titles": [],
            }
        ],
        "failed_gaps": [],
    }

    class TimeoutLLM:
        def call(self, **kwargs):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: TimeoutLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "find_latest_batch_validation_summary",
        lambda project, papers=None: batch_summary,
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(
        papers,
        project="saved-project",
        compact=True,
    )

    assert result["status"] == "failed"
    assert result["error"] == "Experiment suggestion LLM call timed out."
    assert result["validation_used"] is True
    assert result["included_gap_count"] == 1


def test_compact_prompt_for_10_plus_papers_uses_gap_analysis_and_metadata(tmp_path, monkeypatch):
    papers = _papers(13)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(papers, compact=True)

    user_prompt = FakeLLM.calls[0]["user"]
    assert result["paper_count"] == 13
    assert "--- ACTIVE PAPER SET ---" in user_prompt
    assert "Title paper-12" in user_prompt
    assert "research_problem" not in user_prompt
    assert len(user_prompt) < 12000


def test_cache_miss_generates_and_saves_gap_analysis_before_suggesting(tmp_path, monkeypatch):
    papers = _papers(2)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_profile_or_insights",
        lambda source, paper_id: {"paper_id": paper_id, "source": source},
    )
    monkeypatch.setattr(
        experiment_suggester,
        "detect_gaps",
        lambda profiles: (analysis, "{}"),
    )
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(papers, compact=True)

    assert result["gap_source"] == "generated"
    assert result["gap_analysis_path"]
    assert os.path.exists(result["gap_analysis_path"])


def test_timeout_returns_clear_error_without_subset_fallback(tmp_path, monkeypatch):
    papers = _papers(13)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    cached_path = _write_gap_artifact(tmp_path, papers, analysis)

    class TimeoutLLM:
        def call(self, **kwargs):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: TimeoutLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(papers, compact=True)

    assert result["error"] == "Experiment suggestion LLM call timed out."
    assert result["gap_analysis_path"] == str(cached_path)
    assert result["paper_count"] == 13
    assert result["subset_used"] is False
    assert result["experiments"] == []


def test_builds_on_ids_are_filtered_to_active_paper_set(tmp_path, monkeypatch):
    papers = _papers(2)
    analysis = _gap_analysis([p["paper_id"] for p in papers])
    _write_gap_artifact(tmp_path, papers, analysis)
    FakeLLM.calls = []

    monkeypatch.setattr(experiment_suggester, "_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(experiment_suggester, "LLMClient", lambda: FakeLLM())
    monkeypatch.setattr(
        experiment_suggester,
        "load_paper_metadata",
        lambda source, paper_id: {"title": f"Title {paper_id}", "year": "2026"},
    )

    result = experiment_suggester.suggest_experiments_for_papers(papers, compact=True)

    assert result["experiments"][0]["builds_on"] == ["paper-0"]
