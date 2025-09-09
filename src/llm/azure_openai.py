from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; ignore if unavailable
    pass


class AzureOpenAIClient:
    """Thin wrapper for Azure OpenAI Chat Completions via REST.

    Env vars required:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION (e.g., 2024-12-01-preview)
      - AZURE_OPENAI_DEPLOYMENT (deployment name)
      - USE_AZURE_OPENAI ('1' to enable)
    """

    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.enabled = all([self.endpoint, self.api_key, self.api_version, self.deployment]) and os.getenv("USE_AZURE_OPENAI", "0") == "1"
        # Short, configurable timeout to avoid blocking when Azure is unreachable
        try:
            self.timeout = float(os.getenv("AZURE_OPENAI_TIMEOUT", "2"))
        except Exception:
            self.timeout = 2.0

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512, top_p: float = 1.0) -> Optional[str]:
        if not self.enabled:
            return None

        import requests

        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            return content
        except Exception:
            return None


class AzureLLM:
    """Compatibility layer used by Coordinator.

    Provides .enabled and .classify_plan(question) using AzureOpenAIClient.
    """

    def __init__(self) -> None:
        self.client = AzureOpenAIClient()
        self.enabled = self.client.enabled

    def classify_plan(self, question: str) -> List[str]:
        if not self.enabled:
            return []
        sys_msg = {
            "role": "system",
            "content": (
                "You are a planner for a multi-agent system. Available steps are: "
                "memory, research, analysis. For the given question, return ONLY a JSON array "
                "of steps (lowercase strings) in execution order. Example: [\"research\", \"analysis\"]."
            ),
        }
        user_msg = {"role": "user", "content": f"Question: {question}\nPlan:"}
        out = self.client.chat([sys_msg, user_msg], temperature=0.0, max_tokens=64)
        if not out:
            return []
        return parse_llm_plan(out)


# ---- Planning utilities ----
ALLOWED_STEPS = ("memory", "research", "analysis")


def _extract_json_array(text: str) -> Optional[str]:
    """Extract the first JSON array substring from text.

    Supports plain text, markdown code fences (```json ... ```), and text with
    surrounding commentary. Returns the raw substring including brackets, or None.
    """
    if not text:
        return None

    # Common case: fenced code block with json
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1)

    # Fallback: find first [...] balanced-ish slice
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def parse_llm_plan(text: str) -> List[str]:
    """Parse and validate an LLM planning output into a list of steps.

    Rules:
    - Must decode to a JSON array of strings.
    - Allowed values: memory, research, analysis (case-insensitive).
    - Deduplicate while preserving order.
    - Max 3 steps; invalid entries are dropped.
    - Returns [] when parsing/validation fails (caller should fallback).
    """
    raw = _extract_json_array(text)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    normalized: List[str] = []
    seen = set()
    for item in data:
        if not isinstance(item, str):
            continue
        step = item.strip().lower()
        if step in ALLOWED_STEPS and step not in seen:
            normalized.append(step)
            seen.add(step)
        if len(normalized) >= 3:
            break

    return normalized
