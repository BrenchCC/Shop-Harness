"""LLM 客户端层:统一消息模型 + Mock/OpenAI-compatible 两种实现。"""

from .base import LLMClient, LLMError, Message, ToolCall

__all__ = ["LLMClient", "LLMError", "Message", "ToolCall"]
