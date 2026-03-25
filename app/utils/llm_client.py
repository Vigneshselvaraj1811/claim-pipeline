"""
LLM client using Groq with vision support via base64 images.
"""
from __future__ import annotations

import base64
from typing import Optional

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from groq import Groq
        self._client = Groq(api_key=self.settings.groq_api_key)
        return self._client

    def complete(
        self,
        system_prompt: str,
        user_text: str,
        images: Optional[list[bytes]] = None,
    ) -> str:
        client = self._get_client()

        if images:
            # Use vision model with base64 images
            content = []
            for img_bytes in images:
                b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                })
            content.append({"type": "text", "text": user_text})

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # vision model
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": content},
                ],
            )
        else:
            # Text-only fallback
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_text},
                ],
            )

        return response.choices[0].message.content


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client