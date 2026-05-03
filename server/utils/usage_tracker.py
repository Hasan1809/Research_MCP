import json
import os
from datetime import datetime
from utils.logger import get_logger, get_session_dir

logger = get_logger(__name__)


def _normalize_tool_name(tool_name: str) -> str:
    prefix = os.environ.get("USAGE_TOOL_PREFIX", "").strip()
    if prefix and not tool_name.startswith(prefix):
        return f"{prefix}{tool_name}"
    return tool_name

# IONOS pricing per 1M tokens — https://cloud.ionos.com/ai/ai-model-hub
_PRICING = {
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input": 0.35,
        "output": 0.45,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "input": 0.10,
        "output": 0.10,
    },
}
_DEFAULT_PRICING = {"input": 0.50, "output": 0.50}


def log_usage(
    tool_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_seconds: float,
    input_chars: int = 0,
    paper_id: str = "",
):
    normalized_tool_name = _normalize_tool_name(tool_name)
    usage_log_path = os.path.join(get_session_dir(), "usage.log")
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": normalized_tool_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_chars": input_chars,
        "latency_seconds": round(latency_seconds, 2),
        "cost_usd": round(total_cost, 6),
        "paper_id": paper_id,
    }

    os.makedirs(os.path.dirname(usage_log_path), exist_ok=True)
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "USAGE | tool=%s model=%s in=%d out=%d total=%d latency=%.1fs cost=$%.4f",
        normalized_tool_name, model.split("/")[-1],
        input_tokens, output_tokens, total_tokens,
        latency_seconds, total_cost,
    )


def get_usage_summary() -> dict:
    usage_log_path = os.path.join(get_session_dir(), "usage.log")

    if not os.path.exists(usage_log_path):
        return {"total_calls": 0, "total_tokens": 0, "total_cost_usd": 0}

    records = []
    with open(usage_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"total_calls": 0, "total_tokens": 0, "total_cost_usd": 0}

    total_input = sum(r["input_tokens"] for r in records)
    total_output = sum(r["output_tokens"] for r in records)
    total_cost = sum(r["cost_usd"] for r in records)
    total_latency = sum(r["latency_seconds"] for r in records)

    by_tool = {}
    for r in records:
        tool = r["tool"]
        if tool not in by_tool:
            by_tool[tool] = {"calls": 0, "tokens": 0, "cost_usd": 0, "latency": 0}
        by_tool[tool]["calls"] += 1
        by_tool[tool]["tokens"] += r["total_tokens"]
        by_tool[tool]["cost_usd"] = round(by_tool[tool]["cost_usd"] + r["cost_usd"], 6)
        by_tool[tool]["latency"] = round(by_tool[tool]["latency"] + r["latency_seconds"], 1)

    return {
        "total_calls": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_cost_usd": round(total_cost, 4),
        "total_latency_seconds": round(total_latency, 1),
        "avg_latency_seconds": round(total_latency / len(records), 1),
        "by_tool": by_tool,
        "model": records[-1]["model"] if records else "",
    }
