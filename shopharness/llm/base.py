"""统一消息模型与 LLMClient 协议。

Harness 只依赖本模块定义的协议,不关心底层是 Mock 还是 vLLM。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """对齐 OpenAI chat completions 的消息结构。"""

    role: str  # system / user / assistant / tool
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None,
                  tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, name: str, content: str) -> "Message":
        return cls(role="tool", tool_call_id=tool_call_id, name=name, content=content)


class LLMError(RuntimeError):
    """LLM 服务调用失败。"""


class LLMClient(Protocol):
    """模型客户端协议:给定消息历史与可用工具 schema,返回一条 assistant 消息。"""

    def chat(self, messages: list[Message],
             tools: list[dict[str, Any]] | None = None) -> Message:
        ...
