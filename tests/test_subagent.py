"""子代理:委托调用、上下文隔离、摘要回传。"""

from __future__ import annotations

from conftest import tool_calls


def test_delegate_research_via_mock(harness):
    result = harness.handle("YX-1001 和 YX-1003 对比哪个好")
    assert "delegate_research" in tool_calls(result)
    assert not result.handed_off


def test_delegate_aftersale_via_mock(harness):
    result = harness.handle("帮我处理退货,订单 20260701002")
    assert "delegate_aftersale" in tool_calls(result)


def test_context_isolation(harness):
    """子代理的中间工具消息不得进入主对话上下文。"""
    harness.handle("YX-1001 和 YX-1003 对比哪个好")
    main_tool_msgs = [m for m in harness.context.history if m.role == "tool"]
    # 主上下文只有 delegate_* 一条工具结果,子代理的 search_products 不可见
    names = [m.name for m in main_tool_msgs]
    assert "search_products" not in names
    assert names == ["delegate_research"]


def test_subagent_returns_summary_only(harness):
    """回传给主上下文的是摘要,而非完整检索噪音。"""
    harness.handle("YX-1001 和 YX-1003 对比哪个好")
    tool_msg = next(m for m in harness.context.history
                    if m.role == "tool" and m.name == "delegate_research")
    assert "summary" in tool_msg.content
    assert len(tool_msg.content) < 600  # 摘要级长度


def test_subagent_tool_whitelist_enforced(harness):
    """子代理内部不允许调用危险工具。"""
    from shopharness.core.subagent import AFTERSALE_TOOLS, RESEARCH_TOOLS
    assert "adjust_price" not in RESEARCH_TOOLS
    assert "adjust_price" not in AFTERSALE_TOOLS


def test_subagent_trace(harness):
    harness.handle("YX-1001 和 YX-1003 对比哪个好")
    content = harness.tracer.path.read_text(encoding="utf-8")
    assert "subagent_call" in content
    assert "subagent_result" in content
