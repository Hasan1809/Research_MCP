import io
import json
import os
import httpx
from pypdf import PdfReader
from utils.logger import get_logger

logger = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "papers")


def load_cached(source: str, paper_id: str) -> dict | None:
    path = os.path.join(_DATA_DIR, source, f"{paper_id}.json")
    if not os.path.exists(path):
        return None
    logger.info("Loading cached paper: %s", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cached(source: str, paper_id: str, data: dict):
    folder = os.path.join(_DATA_DIR, source)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{paper_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved paper to cache: %s", path)


def download_and_extract_text(pdf_url: str) -> str:
    logger.info("Downloading PDF: %s", pdf_url)
    response = httpx.get(pdf_url, follow_redirects=True, timeout=30)
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
