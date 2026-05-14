import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services import project_manager


def test_batch_add_multiple_mixed_source_papers(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "_PROJECTS_DIR", str(tmp_path))

    result = project_manager.batch_add_papers_to_project(
        "Mixed Project",
        [
            {"paper_id": "2603.17419", "source": "arxiv"},
            {"paper_id": "6aba9c3fc42286af5e8d42712c07a1033a763cc2", "source": "semantic_scholar"},
        ],
    )

    assert result["project"] == "mixed-project"
    assert result["summary"]["added_count"] == 2
    assert result["summary"]["duplicate_count"] == 0
    assert result["summary"]["failed_count"] == 0

    manifest = json.loads((tmp_path / "mixed-project.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "mixed-project"
    assert len(manifest["papers"]) == 2
    assert set(manifest["papers"][0].keys()) == {"paper_id", "source", "added"}


def test_batch_add_detects_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "_PROJECTS_DIR", str(tmp_path))
    project_manager.create_project("Dup Project")
    project_manager.add_paper_to_project("dup-project", "2505.24019", "arxiv")

    result = project_manager.batch_add_papers_to_project(
        "dup-project",
        [
            {"paper_id": "2505.24019", "source": "arxiv"},
            {"paper_id": "7410676678825c855377b7489ac2ee412375908d", "source": "semantic_scholar"},
        ],
    )

    assert result["summary"]["added_count"] == 1
    assert result["summary"]["duplicate_count"] == 1
    assert result["duplicates"][0]["reason"] == "already in project"

    manifest = json.loads((tmp_path / "dup-project.json").read_text(encoding="utf-8"))
    assert len(manifest["papers"]) == 2


def test_batch_add_skips_invalid_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "_PROJECTS_DIR", str(tmp_path))

    result = project_manager.batch_add_papers_to_project(
        "Invalid Refs",
        [
            {"paper_id": "", "source": "arxiv"},
            {"paper_id": "2603.17419"},
            "not-a-dict",
        ],
    )

    assert result["summary"]["added_count"] == 0
    assert result["summary"]["skipped_count"] == 3
    assert result["summary"]["failed_count"] == 0


def test_existing_add_paper_to_project_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "_PROJECTS_DIR", str(tmp_path))
    project_manager.create_project("Single Add")

    manifest = project_manager.add_paper_to_project("single-add", "2603.17419", "arxiv")

    assert manifest["name"] == "single-add"
    assert manifest["papers"][0]["paper_id"] == "2603.17419"
    assert manifest["papers"][0]["source"] == "arxiv"
    assert set(manifest["papers"][0].keys()) == {"paper_id", "source", "added"}
