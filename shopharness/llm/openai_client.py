"""通用云端 Chat Completions 客户端 / Cloud API client."""

from __future__ import annotations

import os
import sys
import json
from typing import Any

from openai import OpenAI

sys.path.append(os.getcwd())

from ..llm.base import LLMError, Message, ToolCall


class OpenAIClient:
    mode = "云端 API"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 120.0,
    ):
        """Initialize with base_url, model, api_key and timeout in seconds."""
        self.model = model
        self.client = OpenAI(
            base_url = base_url,
            api_key = api_key,
            timeout = timeout,
        )

    def _request_options(self) -> dict[str, Any]:
        """Return provider-specific options; cloud requests have none."""
        return {}

    def chat(self, messages: list[Message],
             tools: list[dict[str, Any]] | None = None) -> Message:
        """Send messages and optional function schemas in tools."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_openai(m) for m in messages],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        kwargs.update(self._request_options())
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # 不回显服务端响应或凭据 / Do not expose response bodies or credentials.
            status = getattr(exc, "status_code", None)
            detail = f"HTTP {status}" if isinstance(status, int) else type(exc).__name__
            raise LLMError(f"LLM 请求失败 ({detail})") from None
        try:
            choice = resp.choices[0].message
            tool_calls = None
            if choice.tool_calls:
                tool_calls = [
                    ToolCall(
                        id = tc.id,
                        name = tc.function.name,
                        arguments = json.loads(tc.function.arguments or "{}"),
                    )
                    for tc in choice.tool_calls
                ]
            return Message.assistant(content = choice.content, tool_calls = tool_calls)
        except (ValueError, TypeError, AttributeError, IndexError):
            raise LLMError("LLM 响应格式无效") from None

    @staticmethod
    def _to_openai(m: Message) -> dict[str, Any]:
        """Serialize m, including assistant calls and tool result IDs."""
        msg: dict[str, Any] = {"role": m.role}
        if m.role == "tool":
            msg["tool_call_id"] = m.tool_call_id
            msg["content"] = m.content or ""
            return msg
        msg["content"] = m.content or ""
        if m.tool_calls:
            msg["tool_calls"] = [{
                "id": call.id, "type": "function",
                "function": {"name": call.name,
                             "arguments": json.dumps(call.arguments,
                                                     ensure_ascii=False)},
            } for call in m.tool_calls]
        return msg
