"""上下文工程:三级 compaction 各自触发、事实保留。"""

from __future__ import annotations

from shopharness.config import Settings
from shopharness.core.context import ContextManager
from shopharness.llm.base import Message
from shopharness.llm.mock_client import MockLLM


def make_context(**overrides) -> ContextManager:
    return ContextManager(Settings(**overrides))


def test_level1_tool_result_slimming():
    ctx = make_context(keep_recent_tool_results=1, tool_result_slim_chars=100)
    ctx.history.append(Message.user("查订单"))
    for i in range(3):
        ctx.history.append(Message.tool(f"id-{i}", "get_order", "x" * 500))
    events = ctx.compact(MockLLM())
    assert any(e.level == 1 for e in events)
    archived = [m for m in ctx.history
                if m.role == "tool" and m.content.startswith("[已归档]")]
    assert len(archived) == 2  # 最近 1 条受保护
    # 原文可通过 artifact 引用取回
    assert ctx.artifacts.get("artifact-1") is not None


def test_level2_window_with_fact_extraction():
    ctx = make_context(context_budget=100, keep_recent_turns=4,
                       keep_recent_tool_results=4)
    ctx.history.append(Message.user("你好"))
    ctx.history.append(Message.assistant("您好"))
    for i in range(10):
        ctx.history.append(Message.user(f"我想买 YX-100{i % 3} 第{i}条" + "长" * 50))
        ctx.history.append(Message.assistant("回复" + "复" * 50))
    events = ctx.compact(MockLLM())
    assert any(e.level == 2 for e in events)
    assert "看中的商品" in ctx.state.facts
    assert "YX-1001" in ctx.state.facts["看中的商品"]


def test_level3_full_summary():
    ctx = make_context(context_budget=50, keep_recent_turns=6,
                       keep_tail_after_summary=2, keep_recent_tool_results=6)
    for i in range(12):
        ctx.history.append(Message.user("消息" + "长" * 40))
    events = ctx.compact(MockLLM())
    assert any(e.level == 3 for e in events)
    assert ctx.history[0].content.startswith("[早前会话摘要]")
    assert len(ctx.history) == 3  # 摘要 + 最近 2 条


def test_no_compaction_under_budget():
    ctx = make_context(context_budget=100000)
    ctx.history.append(Message.user("你好"))
    assert ctx.compact(MockLLM()) == []


def test_facts_parse_format():
    from shopharness.core.context import _parse_fact_lines
    facts = _parse_fact_lines("看中的商品: YX-1001\n无\n价格承诺: 900 元")
    assert facts == {"看中的商品": "YX-1001", "价格承诺": "900 元"}


def test_build_injects_state_and_skills():
    from shopharness.core.skills import Skill
    ctx = make_context()
    ctx.state.facts["看中的商品"] = "YX-1001"
    skill = Skill(name="s", intents=[], tools=[], description="",
                  instructions="技能指令正文")
    messages = ctx.build([skill])
    system = messages[0].content
    assert "技能指令正文" in system
    assert "YX-1001" in system  # L2 状态渲染进 system
