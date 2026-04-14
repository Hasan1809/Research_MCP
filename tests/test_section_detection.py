"""
Tests for the section detection heuristics in pdf_service.py.
Run with: python -m pytest tests/test_section_detection.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from services.documents.pdf_service import _is_heading, _heading_level, detect_sections_from_text


# ── _is_heading ──────────────────────────────────────────────────────────────

def test_known_keyword_exact():
    assert _is_heading("Abstract")
    assert _is_heading("Introduction")
    assert _is_heading("Conclusion")
    assert _is_heading("References")
    assert _is_heading("Limitations")
    assert _is_heading("Related Work")

def test_known_keyword_case_insensitive():
    assert _is_heading("INTRODUCTION")
    assert _is_heading("abstract")

def test_numbered_section():
    assert _is_heading("1. Introduction")
    assert _is_heading("3.1 Methods")
    assert _is_heading("2.3.1 Dataset Details")
    assert _is_heading("4. Experimental Setup")

def test_number_then_keyword():
    assert _is_heading("2 Related Work")
    assert _is_heading("3 Conclusion")

def test_all_caps_heading():
    assert _is_heading("RELATED WORK")
    assert _is_heading("METHODOLOGY")

def test_not_heading_ends_with_period():
    assert not _is_heading("This is a normal sentence.")
    assert not _is_heading("The results show significant improvement.")

def test_not_heading_too_long():
    long_line = "This is a very long line that exceeds eighty characters and should not be detected as a heading"
    assert not _is_heading(long_line)

def test_not_heading_empty():
    assert not _is_heading("")
    assert not _is_heading("   ")

def test_not_heading_generic_short_line():
    # Short lines that are not headings
    assert not _is_heading("yes")  # too short / not a keyword
    assert not _is_heading("Table 1")  # not in keyword set


# ── _heading_level ───────────────────────────────────────────────────────────

def test_heading_level_keyword():
    assert _heading_level("Introduction") == 1
    assert _heading_level("Conclusion") == 1

def test_heading_level_numbered():
    assert _heading_level("1. Introduction") == 1
    assert _heading_level("3.1 Methods") == 2
    assert _heading_level("2.3.1 Details") == 3

def test_heading_level_all_caps():
    assert _heading_level("INTRODUCTION") == 1


# ── detect_sections_from_text ────────────────────────────────────────────────

_SAMPLE_TEXT = """\
A Sample Paper Title

Abstract

This is the abstract of the paper. It summarizes the work.

1. Introduction

This is the introduction section with some text about the problem.
More text about the motivation.

2. Methods

We use a specific method to solve the problem.
Details about the experimental setup follow.

2.1 Dataset

We evaluate on Dataset X and Dataset Y.

3. Results

The results show a 10% improvement over baseline.

4. Conclusion

In this paper we proposed a new method. Future work includes scaling.
"""


def test_basic_section_detection():
    result = detect_sections_from_text(_SAMPLE_TEXT)
    headings = [s["heading"] for s in result["sections"]]
    assert "Abstract" in headings
    assert "1. Introduction" in headings
    assert "2. Methods" in headings
    assert "3. Results" in headings
    assert "4. Conclusion" in headings


def test_abstract_extraction():
    result = detect_sections_from_text(_SAMPLE_TEXT)
    assert "abstract" in result["metadata"]["abstract"].lower()


def test_section_text_not_empty():
    result = detect_sections_from_text(_SAMPLE_TEXT)
    for sec in result["sections"]:
        if sec["heading"] in ("1. Introduction", "2. Methods", "3. Results"):
            assert len(sec["text"]) > 0, f"Section {sec['heading']} has empty text"


def test_no_sections_fallback():
    plain = "This is just some plain text without any headings or structure at all.\nMore text here."
    result = detect_sections_from_text(plain)
    # Should produce at least one section (fallback "Full Text" or "Preamble")
    assert len(result["sections"]) >= 1
    assert result["sections"][0]["text"]


def test_metadata_fields():
    result = detect_sections_from_text(_SAMPLE_TEXT)
    meta = result["metadata"]
    assert "title" in meta
    assert "abstract" in meta
    assert "char_count" in meta
    assert meta["char_count"] == len(_SAMPLE_TEXT)


def test_full_text_preserved():
    result = detect_sections_from_text(_SAMPLE_TEXT)
    assert result["full_text"] == _SAMPLE_TEXT
