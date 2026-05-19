import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from tools import build_paper_profile


def test_large_profile_falls_back_to_truncated_full_text_when_chunks_missing(monkeypatch):
    large_text = "This section discusses LLM agent security and indirect prompt injection. " * 2000
    cached = {
        "full_text": large_text,
        "sections": [{"heading": "Body", "text": large_text}],
    }
    captured = {}

    monkeypatch.setattr(build_paper_profile, "MAX_FULL_TEXT_CHARS", 1000)
    monkeypatch.setattr(build_paper_profile, "load_cached", lambda source, paper_id: cached)
    monkeypatch.setattr(build_paper_profile, "load_profile", lambda source, paper_id: None)
    monkeypatch.setattr(build_paper_profile, "_retrieve_profile_chunks", lambda paper_id, source: [])
    monkeypatch.setattr(build_paper_profile, "save_profile", lambda source, paper_id, result: None)
    monkeypatch.setattr(build_paper_profile, "log_invocation", lambda *_args, **_kwargs: None)

    def fake_build_profile(context_text, paper_id):
        captured["context_text"] = context_text
        return {
            "paper_type": "empirical study",
            "research_problem": "problem",
            "main_contribution": "contribution",
        }, "{}"

    monkeypatch.setattr(build_paper_profile, "build_profile", fake_build_profile)

    result = build_paper_profile.build_paper_profile_tool(
        paper_id="2604.11790",
        source="arxiv",
        force=True,
    )

    assert result["_meta"]["context_path"] == "truncated_full_text"
    assert result["_meta"]["context_chars"] == 1000
    assert len(captured["context_text"]) == 1000
