"""转人工兜底:触发检测 + 交接摘要生成。

不追求 100% 自动化,追求"该转就转、转得漂亮":
摘要让坐席零上下文成本接手,摘要质量本身纳入评测。
"""

from __future__ import annotations

from typing import Any

from .context import SessionState

HANDOFF_KEYWORDS = ["转人工", "人工客服", "真人", "投诉", "找人工"]


def detect_handoff(user_text: str) -> str | None:
    """用户明确要求人工时返回触发原因,否则返回 None。"""
    for kw in HANDOFF_KEYWORDS:
        if kw in user_text:
            return f"买家主动要求({kw})"
    return None


def build_summary(state: SessionState, action_log: list[str],
                  reason: str) -> str:
    """确定性模板生成交接摘要(不依赖模型,保证可复现、可评测)。"""
    lines = [f"【转人工交接摘要】触发原因:{reason}"]
    facts = state.facts
    if state.intent:
        lines.append(f"买家意图:{state.intent}")
    if facts:
        lines.append("已确认事实:")
        for key, value in facts.items():
            lines.append(f"  - {key}: {value}")
    if action_log:
        lines.append("已执行操作:")
        for action in action_log[-10:]:
            lines.append(f"  - {action}")
    else:
        lines.append("已执行操作:无")
    lines.append("建议下一步:人工坐席复核上述事实后继续处理买家诉求。")
    return "\n".join(lines)


def handoff_reply(summary: str, ticket_id: int | None) -> str:
    ticket = f",工单号 #{ticket_id}" if ticket_id else ""
    return (f"抱歉给您带来不便,已为您转接人工客服{ticket}。"
            f"以下是交接摘要,人工坐席将直接接手,无需您重复描述:\n{summary}")
