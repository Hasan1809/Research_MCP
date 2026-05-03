import json
import logging
import os
from datetime import datetime
from typing import Optional

_initialized = False
_tools_dir: str = ""
_session_dir: str = ""
_invocation_counter: int = 0


def init_logging() -> None:
    global _initialized, _tools_dir, _session_dir
    if _initialized:
        return
    _initialized = True

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_dir = os.path.join(logs_dir, f"langchain_session_{timestamp}")
    _tools_dir = os.path.join(_session_dir, "tools")
    os.makedirs(_tools_dir, exist_ok=True)

    log_file = os.path.join(_session_dir, "main.log")
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_session_dir() -> str:
    if not _initialized:
        init_logging()
    return _session_dir


def log_invocation(tool_name: str, arguments: dict, output=None, error: Optional[str] = None) -> None:
    global _invocation_counter
    _invocation_counter += 1
    filename = f"{_invocation_counter:03d}_{tool_name}.json"
    path = os.path.join(_tools_dir, filename)
    record = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "arguments": arguments,
        "output": output,
    }
    if error is not None:
        record["error"] = error
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
