import json
import logging
import os
from datetime import datetime

_initialized = False
_invocation_dir: str = ""
_invocation_counter: int = 0


def init_logging():
    global _initialized, _invocation_dir
    if _initialized:
        return
    _initialized = True

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"session_{timestamp}.log")

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    _invocation_dir = os.path.join(logs_dir, "tool_invocations", f"session_{timestamp}")
    os.makedirs(_invocation_dir, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_invocation(tool_name: str, arguments: dict, output=None, error: str | None = None):
    global _invocation_counter
    _invocation_counter += 1
    filename = f"{_invocation_counter:03d}_{tool_name}.json"
    path = os.path.join(_invocation_dir, filename)
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
