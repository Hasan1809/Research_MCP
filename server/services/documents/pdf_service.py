import io
import re
from typing import List, Optional

import httpx
from pypdf import PdfReader

from config import PDF_DOWNLOAD_TIMEOUT
from services.paper_repository import load_paper_cache, save_paper_cache
from utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_KEYWORDS = frozenset({
    "abstract", "introduction", "related work", "background", "motivation",
    "methodology", "method", "methods", "approach", "proposed method",
    "experimental setup", "experiments", "experiment", "evaluation",
    "results", "discussion", "analysis", "conclusion", "conclusions", "summary",
    "future work", "future directions", "limitations", "acknowledgments",
    "acknowledgements", "references", "appendix", "appendices",
    "dataset", "datasets", "baseline", "baselines", "model", "architecture",
    "framework", "system overview", "overview", "problem formulation",
    "problem statement", "preliminaries", "notation",
})

_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s\-:]{3,}$")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or stripped.endswith("."):
        return False
    if _NUMBERED_RE.match(stripped):
        return True
    lower = stripped.lower()
    if lower in _SECTION_KEYWORDS:
        return True
    parts = lower.split(None, 1)
    if len(parts) == 2 and parts[0].rstrip(".").isdigit() and parts[1] in _SECTION_KEYWORDS:
        return True
    if _ALL_CAPS_RE.match(stripped):
        return True
    return False


def _heading_level(heading: str) -> int:
    match = _NUMBERED_RE.match(heading.strip())
    if match:
        num_part = heading.strip().split()[0].rstrip(".")
        return len(num_part.split("."))
    return 1


def detect_sections_from_text(text: str, pages_text: Optional[List[str]] = None) -> dict:
    page_offsets: list[int] = []
    if pages_text:
        offset = 0
        for pt in pages_text:
            page_offsets.append(offset)
            offset += len(pt) + 1

    lines = text.split("\n")
    sections: list[dict] = []
    current_heading = "Preamble"
    current_level = 1
    body_lines: list[str] = []
    heading_char_offset = 0
    char_offset = 0

    def _char_to_page(offset: int) -> int:
        if not page_offsets:
            return 0
        for i in range(len(page_offsets) - 1, -1, -1):
            if offset >= page_offsets[i]:
                return i
        return 0

    def _flush(heading: str, level: int, body: list[str], h_offset: int) -> None:
        text_body = "\n".join(body).strip()
        if text_body or heading != "Preamble":
            sections.append({
                "heading": heading,
                "level": level,
                "text": text_body,
                "start_page": _char_to_page(h_offset),
                "end_page": _char_to_page(char_offset),
            })

    for line in lines:
        if _is_heading(line.strip()):
            _flush(current_heading, current_level, body_lines, heading_char_offset)
            current_heading = line.strip()
            current_level = _heading_level(current_heading)
            body_lines = []
            heading_char_offset = char_offset
        else:
            body_lines.append(line)
        char_offset += len(line) + 1

    _flush(current_heading, current_level, body_lines, heading_char_offset)

    if not sections:
        sections = [{
            "heading": "Full Text",
            "level": 1,
            "text": text.strip(),
            "start_page": 0,
            "end_page": _char_to_page(char_offset),
        }]

    title = _extract_title(text)
    abstract = _extract_abstract(sections)

    logger.info(
        "Section detection: %d sections, title=%r, abstract_len=%d",
        len(sections), title[:60] if title else "", len(abstract),
    )

    return {
        "full_text": text,
        "sections": sections,
        "metadata": {
            "title": title,
            "abstract": abstract,
            "page_count": len(pages_text) if pages_text else 0,
            "char_count": len(text),
        },
    }


def _extract_title(text: str) -> str:
    for line in text.split("\n")[:30]:
        stripped = line.strip()
        if len(stripped) > 20 and not stripped[0].isdigit():
            return stripped
    return ""


def _extract_abstract(sections: list[dict]) -> str:
    for sec in sections:
        if "abstract" in sec["heading"].lower():
            return sec["text"]
    return ""


def load_cached(source: str, paper_id: str) -> Optional[dict]:
    paper = load_paper_cache(source, paper_id)
    if paper is None:
        return None
    logger.info("Loading cached paper: data/papers/%s/%s.json", source, paper_id)
    return paper


def save_cached(source: str, paper_id: str, data: dict):
    save_paper_cache(source, paper_id, data)
    logger.info("Saved paper to cache: data/papers/%s/%s.json", source, paper_id)


def download_and_extract_text(pdf_url: str) -> str:
    logger.info("Downloading PDF: %s", pdf_url)
    response = httpx.get(pdf_url, follow_redirects=True, timeout=PDF_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    logger.info("PDF downloaded: %d bytes", len(response.content))

    try:
        reader = PdfReader(io.BytesIO(response.content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        logger.info("Text extracted: %d characters from %d pages", len(text), len(reader.pages))
        return text
    except Exception as e:
        logger.exception("PDF text extraction failed: %s", e)
        raise


def extract_structured_text(pdf_url: str) -> dict:
    logger.info("Downloading PDF for structured extraction: %s", pdf_url)
    response = httpx.get(pdf_url, follow_redirects=True, timeout=PDF_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    logger.info("PDF downloaded: %d bytes", len(response.content))

    try:
        reader = PdfReader(io.BytesIO(response.content))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text)
        logger.info("Text extracted: %d chars from %d pages", len(full_text), len(pages_text))
        return detect_sections_from_text(full_text, pages_text)
    except Exception as e:
        logger.exception("Structured PDF extraction failed: %s", e)
        raise
