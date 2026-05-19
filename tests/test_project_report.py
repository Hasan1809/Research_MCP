import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.reports import project_report


def test_generate_project_report_from_latest_artifacts(tmp_path, monkeypatch):
    project = "report-test"
    papers = [{"paper_id": "p1", "source": "arxiv"}]
    manifest = {"name": project, "created": "2026-05-19T00:00:00", "papers": papers}

    analysis_dir = tmp_path / "analysis"
    validation_dir = analysis_dir / "gap_validations" / "batches"
    bibliography_dir = tmp_path / "artifacts" / "bibliographies"
    report_dir = tmp_path / "artifacts" / "reports"
    analysis_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    bibliography_dir.mkdir(parents=True)

    gap_path = analysis_dir / "gap_analysis_20260519_test.json"
    gap_path.write_text(
        json.dumps({
            "papers": papers,
            "analysis": {
                "research_gaps": [{"gap": "Research gap one", "evidence": "Evidence"}],
                "methodological_gaps": [{"gap": "Method gap one", "missing_approach": "Missing"}],
                "contradictions": [{}],
                "connections": [{}],
            },
        }),
        encoding="utf-8",
    )

    validation_path = validation_dir / "batch_gap_validation_20260519_report-test.json"
    validation_path.write_text(
        json.dumps({
            "project": project,
            "project_papers": papers,
            "validated_gaps": [
                {
                    "original_gap": "Research gap one",
                    "status": "partially_addressed",
                    "confidence": "medium",
                    "use_for_experiments": True,
                    "refined_gap": "Refined research gap one",
                    "decision_reason": "Partial reason",
                },
                {
                    "original_gap": "Closed gap",
                    "status": "already_addressed",
                    "confidence": "high",
                    "use_for_experiments": False,
                    "decision_reason": "Closed reason",
                    "external_evidence_titles": ["Evidence title"],
                    "evidence_summary": [
                        {
                            "title": "Evidence title",
                            "year": "2026",
                            "source": "arxiv",
                            "classification": "directly_addresses_gap",
                            "score": 0.91,
                            "reason": "Direct reason",
                        }
                    ],
                },
            ],
            "batch_artifact_path": str(validation_path),
        }),
        encoding="utf-8",
    )

    experiments_path = analysis_dir / "experiments_20260519_test.json"
    experiments_path.write_text(
        json.dumps({
            "papers": papers,
            "experiments": {
                "experiments": [
                    {
                        "title": "Experiment A",
                        "hypothesis": "Hypothesis",
                        "method": "Method",
                        "feasibility": "high",
                        "builds_on": ["p1"],
                        "addresses_gap": "Refined research gap one",
                    }
                ]
            },
        }),
        encoding="utf-8",
    )

    bib_path = bibliography_dir / "report-test_20260519_000000.bib"
    bib_path.write_text("@article{p1,\n  title = {Paper One},\n}", encoding="utf-8")

    monkeypatch.setattr(project_report, "_ANALYSIS_DIR", analysis_dir)
    monkeypatch.setattr(project_report, "_VALIDATION_BATCH_DIR", validation_dir)
    monkeypatch.setattr(project_report, "_BIBLIOGRAPHY_DIR", bibliography_dir)
    monkeypatch.setattr(project_report, "_REPORT_DIR", report_dir)
    monkeypatch.setattr(project_report, "get_project", lambda name: manifest)
    monkeypatch.setattr(
        project_report,
        "load_paper_metadata",
        lambda source, paper_id: {
            "paper_id": paper_id,
            "source": source,
            "title": "Paper One",
            "year": "2026",
        },
    )

    result = project_report.generate_project_report(project)

    assert result["project"] == project
    assert result["paper_count"] == 1
    assert result["gap_count"] == 2
    assert result["included_validated_gap_count"] == 1
    assert result["excluded_validated_gap_count"] == 1
    assert result["experiment_count"] == 1
    assert os.path.exists(result["report_path"])

    report = open(result["report_path"], encoding="utf-8").read()
    assert "# Research Agent Report" in report
    assert "## Papers Included" in report
    assert "Paper One - 2026 - arxiv - p1" in report
    assert "## Gap Detection Summary" in report
    assert "## Gap Validation Summary" in report
    assert "Included validated gaps: 1" in report
    assert "Excluded validated gaps: 1" in report
    assert "Evidence title - 2026 - arxiv - directly_addresses_gap - 0.91" in report
    assert "## Suggested Experiments" in report
    assert "Experiment A" in report
    assert "## Bibliography" in report
