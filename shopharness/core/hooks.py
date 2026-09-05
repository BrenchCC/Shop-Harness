"""pre/post tool hook 总线。

- pre hook:工具执行前拦截,可拒绝(护栏场景,如价格篡改检测)
- post hook:工具执行后回调(审计、埋点)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HookResult:
    allow: bool = True
    reason: str | None = None

    @classmethod
    def deny(cls, reason: str) -> "HookResult":
        return cls(allow=False, reason=reason)


PreHook = Callable[[str, dict[str, Any]], HookResult]
PostHook = Callable[[str, dict[str, Any], dict[str, Any]], None]


@dataclass
class HookBus:
    pre_hooks: list[PreHook] = field(default_factory=list)
    post_hooks: list[PostHook] = field(default_factory=list)

    def run_pre(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """任一 pre hook 拒绝则整体拒绝(短路)。"""
        for hook in self.pre_hooks:
            result = hook(tool_name, args)
            if not result.allow:
                return result
        return HookResult()

    def run_post(self, tool_name: str, args: dict[str, Any],
                 result: dict[str, Any]) -> None:
        for hook in self.post_hooks:
            hook(tool_name, args, result)
