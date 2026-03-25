"""
Robust JSON extraction from LLM responses that may include markdown fences.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    """
    Try multiple strategies to parse JSON from an LLM response string.
    Raises ValueError if no valid JSON found.
    """
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first {...} block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")
