"""分层记忆:画像更新、摘要沉淀、L1 注入、偏好蒸馏。"""

from __future__ import annotations

import pytest

from shopharness.core.memory import MemoryStore
from shopharness.data.seed import ensure_db


@pytest.fixture()
def store(settings):
    conn = ensure_db(settings.db_path)
    return MemoryStore(conn)


def test_profile_update_dedup(store):
    store.update_profile("b1", ["价格敏感", "关注品类:数码影音"])
    store.update_profile("b1", ["价格敏感", "有改价行为"])
    profile = store.get_profile("b1")
    assert profile.count("价格敏感") == 1
    assert "有改价行为" in profile
    assert store.get_profile("nobody") == ""


def test_session_summaries_recent(store):
    for i in range(5):
        store.add_session_summary("b1", f"摘要{i}")
    recent = store.recent_summaries("b1", limit=3)
    assert recent == ["摘要4", "摘要3", "摘要2"]


def test_build_l1_context(store):
    assert store.build_l1_context("nobody") == ""
    store.update_profile("b1", ["价格敏感"])
    store.add_session_summary("b1", "上次咨询了耳机")
    l1 = store.build_l1_context("b1")
    assert "价格敏感" in l1 and "上次咨询了耳机" in l1


def test_distill_preferences(store):
    prefs = store.distill_preferences(
        {"看中的商品": "YX-1001", "价格承诺": "900 元"},
        ["adjust_price(...)"], handed_off=True)
    assert "价格敏感" in prefs
    assert "有改价行为" in prefs
    assert "有转人工历史" in prefs
    assert "关注品类:数码影音" in prefs


def test_harness_memory_injection_and_end_session(settings, mock):
    """端到端:第一轮注入记忆 → 会话结束沉淀 → 新会话再次注入。"""
    from shopharness.cli import build_harness
    from shopharness.core.memory import MemoryStore
    from shopharness.data.seed import ensure_db

    # 预置记忆
    conn = ensure_db(settings.db_path)
    store = MemoryStore(conn)
    store.update_profile("b1", ["价格敏感"])

    h1 = build_harness(settings, mock, buyer_id="b1")
    from conftest import event_types
    r = h1.handle("有耳机推荐吗")
    assert "memory_injected" in event_types(r)
    system = h1.context.build([])[0].content
    assert "价格敏感" in system  # L1 进入 system prompt
    h1.end_session()

    # 新会话应看到上次沉淀的摘要
    h2 = build_harness(settings, mock, buyer_id="b1")
    r2 = h2.handle("你好")
    assert "memory_injected" in event_types(r2)
    l1 = h2.context.l1_context
    assert "近期会话" in l1 and "摘要" in l1


def test_end_session_writes_summary(settings, mock):
    from shopharness.cli import build_harness
    h = build_harness(settings, mock, buyer_id="b2")
    h.handle("有耳机推荐吗")
    h.end_session()
    rows = h.conn.execute(
        "SELECT summary FROM session_summaries WHERE buyer_id='b2'").fetchall()
    assert len(rows) == 1 and "摘要" in rows[0][0]
