import json
import time

import httpx

from config import IONOS_API_TOKEN, IONOS_BASE_URL, IONOS_MODEL, LLM_TEMPERATURE
from utils.usage_tracker import log_usage


class LLMClient:
    def __init__(self, model: str = IONOS_MODEL):
        self.model = model

    def call(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        timeout: int = 60,
        tool_name: str = "",
        input_chars: int = 0,
        paper_id: str = "",
    ) -> tuple[dict, str]:
        payload = {
            "model": self.model,
            "temperature": LLM_TEMPERATURE,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.time()
        response = httpx.post(
            f"{IONOS_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {IONOS_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        latency = time.time() - start
        resp = response.json()
        usage = resp.get("usage", {})
        log_usage(
            tool_name=tool_name,
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_seconds=latency,
            input_chars=input_chars,
            paper_id=paper_id,
        )
        raw = resp["choices"][0]["message"]["content"].strip()

        from services.extraction.llm_extractor import _strip_code_fences

        return json.loads(_strip_code_fences(raw)), raw
