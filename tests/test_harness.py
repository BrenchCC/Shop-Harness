"""Harness 主 loop:完整 turn、非法工具纠正、熔断、步数上限。"""

from __future__ import annotations

from shopharness.llm.base import Message, ToolCall
from shopharness.llm.mock_client import MockLLM

from conftest import event_types, tool_calls


def test_simple_product_turn(harness):
    result = harness.handle("有耳机推荐吗")
    assert "search_products" in tool_calls(result)
    assert "YX-1001" in result.reply or "YX-1002" in result.reply
    assert not result.handed_off


def test_invalid_tool_correction_then_recover(settings):
    mock = MockLLM(queue=[
        Message.assistant(tool_calls=[
            ToolCall(id="x1", name="delete_order", arguments={})]),
        # 第二轮走规则回复
    ])
    harness = __import__("shopharness.cli", fromlist=["build_harness"]) \
        .build_harness(settings, mock)
    result = harness.handle("帮我查一下订单 20260701001 的状态")
    assert "correction" in event_types(result)
    assert not result.handed_off


def test_repeated_invalid_tool_hands_off(settings):
    mock = MockLLM(queue=[
        Message.assistant(tool_calls=[
            ToolCall(id="x1", name="delete_order", arguments={})]),
        Message.assistant(tool_calls=[
            ToolCall(id="x2", name="hack_price", arguments={})]),
    ])
    from shopharness.cli import build_harness
    harness = build_harness(settings, mock)
    result = harness.handle("随便说点什么")
    assert event_types(result).count("correction") == 2
    assert result.handed_off


def test_circuit_breaker_on_repeated_tool_failure(harness):
    # 订单不存在 → 软失败 ×2 → 熔断 → 转人工
    result = harness.handle("帮我查一下订单 20269999999")
    assert "circuit_break" in event_types(result)
    assert result.handed_off
    assert "连续失败" in result.handoff_summary


def test_max_steps_handoff(settings):
    """模型每步都发起成功但无休止的工具调用 → 步数超限转人工。"""
    from shopharness.config import Settings
    from shopharness.cli import build_harness

    class LoopLLM(MockLLM):
        def chat(self, messages, tools=None):
            return self._call("search_products", {"keyword": "耳机"})

    s = settings.model_copy(update={"max_tool_steps": 3})
    harness = build_harness(s, LoopLLM())
    result = harness.handle("有耳机推荐吗")
    assert result.handed_off
    assert "步数超限" in result.handoff_summary


def test_trace_file_written(harness, settings):
    harness.handle("有耳机推荐吗")
    content = harness.tracer.path.read_text(encoding="utf-8")
    assert "user_message" in content
    assert "tool_result" in content
    assert "gen_ai.system" in content
