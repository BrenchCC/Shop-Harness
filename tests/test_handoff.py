"""转人工:关键词触发、熔断触发、交接摘要内容。"""

from __future__ import annotations

from shopharness.core.context import SessionState
from shopharness.core.handoff import (build_summary, detect_handoff,
                                      handoff_reply)


def test_detect_handoff_keywords():
    assert detect_handoff("我要投诉") is not None
    assert detect_handoff("转人工") is not None
    assert detect_handoff("帮我查下订单") is None


def test_summary_contains_facts_and_actions():
    state = SessionState(intent="改价", facts={"看中的商品": "YX-1001"})
    summary = build_summary(state, ["get_order(20260701001)"], "买家主动要求")
    assert "买家主动要求" in summary
    assert "YX-1001" in summary
    assert "get_order" in summary
    assert "建议下一步" in summary


def test_handoff_reply_with_ticket():
    reply = handoff_reply("摘要", 42)
    assert "#42" in reply and "交接摘要" in reply


def test_harness_handoff_creates_ticket(harness):
    from conftest import event_types
    result = harness.handle("我要投诉,转人工")
    assert result.handed_off
    assert "handoff" in event_types(result)
    assert "工单号" in result.reply
    row = harness.conn.execute(
        "SELECT issue_type, summary FROM tickets").fetchone()
    assert row[0] == "转人工" and "转人工交接摘要" in row[1]


def test_handoff_summary_includes_prior_actions(harness):
    harness.handle("帮我查一下订单 20260701001 的状态")
    result = harness.handle("转人工")
    assert "get_order" in result.handoff_summary
