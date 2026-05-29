import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.citations import _safe_artifact_stem


def test_bibliography_artifact_stem_is_capped_for_windows_paths():
    stem = _safe_artifact_stem(
        "papers, select relevant papers, ingest or retrieve them, build paper profiles, "
        "detect research gaps, validate the gaps where possible, suggest experiments, "
        "generate a bibliography, and generate the final report",
    )

    assert len(stem) <= 80
    assert stem
