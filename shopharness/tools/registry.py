"""工具注册表:schema 注册、权限标注、按技能白名单动态裁剪。

schema 完全采用 OpenAI function calling 格式,与 MCP 工具描述一一对应,
后续迁移到 FastMCP 独立进程时无需改动模型侧。
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.permissions import Level

ToolFn = Callable[..., dict[str, Any]]


class ToolError(RuntimeError):
    """工具执行失败(会被 Harness 计为一次失败并回注给模型)。"""


class Tool:
    def __init__(self, name: str, description: str, parameters: dict[str, Any],
                 level: Level, fn: ToolFn):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.level = level
        self.fn = fn

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.fn(**args)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, whitelist: set[str] | None = None) -> list[dict[str, Any]]:
        """按白名单裁剪工具集(8B 模型上下文有限,只注入当前意图相关工具)。"""
        tools = self._tools.values()
        if whitelist is not None:
            tools = [t for t in tools if t.name in whitelist]
        return [t.schema() for t in tools]


def _props(**kwargs: Any) -> dict[str, Any]:
    return {"type": "object", "properties": kwargs.pop("properties"),
            "required": kwargs.pop("required", [])}
