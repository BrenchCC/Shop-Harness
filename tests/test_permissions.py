"""权限模型:三级权限、危险操作确认门、最低限价护栏。"""

from __future__ import annotations

from shopharness.core.hooks import HookBus
from shopharness.core.permissions import Level, PermissionManager
from shopharness.data.seed import ensure_db
from shopharness.tools.servers import make_price_guardrail


def test_read_write_pass_without_confirmation():
    pm = PermissionManager()
    assert pm.check(Level.READ, "get_order").allowed
    assert pm.check(Level.WRITE, "add_order_note").allowed


def test_dangerous_blocked_until_confirmed():
    pm = PermissionManager()
    decision = pm.check(Level.DANGEROUS, "adjust_price")
    assert not decision.allowed and decision.need_confirm
    pm.confirm("adjust_price")
    assert pm.check(Level.DANGEROUS, "adjust_price").allowed


def test_price_guardrail_denies_below_min_price(settings):
    conn = ensure_db(settings.db_path)
    guard = make_price_guardrail(conn)
    bus = HookBus(pre_hooks=[guard])
    # YX-1001 最低限价 880
    denied = bus.run_pre("adjust_price", {
        "order_id": "20260701001", "new_price": 500.0, "reason": "x"})
    assert not denied.allow and "最低限价" in denied.reason
    allowed = bus.run_pre("adjust_price", {
        "order_id": "20260701001", "new_price": 900.0, "reason": "x"})
    assert allowed.allow


def test_guardrail_ignores_other_tools(settings):
    conn = ensure_db(settings.db_path)
    bus = HookBus(pre_hooks=[make_price_guardrail(conn)])
    assert bus.run_pre("get_order", {"order_id": "x"}).allow


def test_harness_adjust_price_full_drama(harness):
    """端到端:拦截 → 复述确认 → 确认 → 成功改价。"""
    from conftest import event_types
    r1 = harness.handle("帮我把订单 20260701001 改价到 900 元")
    assert "dangerous_intercepted" in event_types(r1)
    assert "确认" in r1.reply
    r2 = harness.handle("确认")
    assert "confirmed" in event_types(r2)
    row = harness.conn.execute(
        "SELECT amount FROM orders WHERE order_id='20260701001'").fetchone()
    assert row[0] == 900.0
    # 审计表有拦截与执行两条记录
    audits = harness.conn.execute(
        "SELECT COUNT(*) FROM audit WHERE tool='adjust_price'").fetchone()
    assert audits[0] >= 2
