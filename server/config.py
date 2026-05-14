import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

FULL_TEXT_CHAR_LIMIT = int(os.environ.get("FULL_TEXT_CHAR_LIMIT", 80_000))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
IONOS_API_TOKEN = os.environ.get("IONOS_API_TOKEN", "")
IONOS_BASE_URL = os.environ.get("IONOS_BASE_URL", "").rstrip("/")
IONOS_MODEL = os.environ.get("IONOS_MODEL", "")
S2_API_KEY = os.environ.get("S2_API_KEY")
