"""OpenAI-compatible 客户端:对接本地 vLLM(Qwen3-8B-FP8)。

vLLM 启动参数见 scripts/serve_vllm.sh(--tool-call-parser hermes
--reasoning-parser qwen3 --enable-auto-tool-choice)。
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from ..llm.base import LLMError, Message, ToolCall


class OpenAIClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen3-8B-FP8",
                 enable_thinking: bool = False, timeout: float = 120.0):
        self.model = model
        self.enable_thinking = enable_thinking
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)

    def chat(self, messages: list[Message],
             tools: list[dict[str, Any]] | None = None) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_openai(m) for m in messages],
            "temperature": 0.3,
            "max_tokens": 1024,
            # Qwen3 混合推理:客服场景默认 /no_think 压低首 token 延迟
            "extra_body": {"chat_template_kwargs":
                           {"enable_thinking": self.enable_thinking}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        choice = resp.choices[0].message
        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name,
                         arguments=json.loads(tc.function.arguments or "{}"))
                for tc in choice.tool_calls
            ]
        return Message.assistant(content=choice.content, tool_calls=tool_calls)

    @staticmethod
    def _to_openai(m: Message) -> dict[str, Any]:
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
