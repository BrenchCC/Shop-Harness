"""独立 vLLM HTTP 客户端 / No vLLM server package required."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.append(os.getcwd())

from .openai_client import OpenAIClient


class VLLMClient(OpenAIClient):
    mode = "vLLM"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-8B-FP8",
        enable_thinking: bool = False,
        timeout: float = 120.0,
    ):
        """Connect to base_url using model, enable_thinking and timeout seconds."""
        self.enable_thinking = enable_thinking
        super().__init__(
            base_url = base_url,
            model = model,
            api_key = "EMPTY",
            timeout = timeout,
        )

    def _request_options(self) -> dict[str, Any]:
        """Return Qwen template options exclusively for the vLLM backend."""
        # 本地 Qwen 默认关闭思考 / Disable thinking for local Qwen by default.
        return {"extra_body": {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }}
