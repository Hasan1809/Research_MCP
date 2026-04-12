import io
import httpx
from pypdf import PdfReader
from utils.logger import get_logger

logger = get_logger(__name__)


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
