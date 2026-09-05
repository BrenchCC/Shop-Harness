"""三级权限模型:READ / WRITE / DANGEROUS。

- READ:直接执行
- WRITE:执行 + 审计落库
- DANGEROUS:确认门——本会话内买家必须先明确确认过该工具,
  否则拦截并让模型先复述确认;防止 8B 模型被诱导直接改价/退款。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Level(enum.Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass
class Decision:
    allowed: bool
    need_confirm: bool = False
    reason: str | None = None


@dataclass
class PermissionManager:
    """按会话维护危险工具的确认状态。"""

    _confirmed: set[str] = field(default_factory=set)

    def check(self, level: Level, tool_name: str) -> Decision:
        if level is Level.DANGEROUS and tool_name not in self._confirmed:
            return Decision(
                allowed=False,
                need_confirm=True,
                reason=(
                    f"[系统拦截] {tool_name} 为危险操作。请先用文字向买家复述"
                    "操作内容(对象、变更值、影响)并请买家明确回复「确认」,"
                    "确认后再调用本工具。"
                ),
            )
        return Decision(allowed=True)

    def confirm(self, tool_name: str) -> None:
        self._confirmed.add(tool_name)

    def is_confirmed(self, tool_name: str) -> bool:
        return tool_name in self._confirmed
