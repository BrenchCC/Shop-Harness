"""售后长流程:节点执行、interrupt、checkpoint 重启恢复。"""

from __future__ import annotations

import pytest

from shopharness.data.seed import ensure_db
from shopharness.flows.aftersale import AftersaleFlow


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "shop.db")
    ensure_db(path)
    return path


def test_flow_start_interrupts_for_buyer_confirmation(db_path):
    flow = AftersaleFlow(db_path)
    view = flow.start("t-1", "20260701002")
    assert view["status"] == "waiting_buyer"
    assert "7 天无理由退换期内" in view["question"]
    assert "极光机械键盘" in view["question"]


def test_flow_resume_approved_creates_ticket(db_path):
    flow = AftersaleFlow(db_path)
    flow.start("t-2", "20260701002")
    view = flow.resume("t-2", "确认,按这个方案来")
    assert view["status"] == "done"
    assert "工单" in view["result"]
    row = flow.conn.execute(
        "SELECT issue_type FROM tickets ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "售后"


def test_flow_resume_rejected(db_path):
    flow = AftersaleFlow(db_path)
    flow.start("t-3", "20260701002")
    view = flow.resume("t-3", "我再想想,先不处理")
    assert view["status"] == "done"
    assert "未确认" in view["result"]


def test_flow_unknown_order_terminates(db_path):
    flow = AftersaleFlow(db_path)
    view = flow.start("t-4", "20000000000")
    assert view["status"] == "done"
    assert "不存在" in view["result"]


def test_checkpoint_recovery_after_restart(db_path):
    """模拟进程重启:新实例能从 interrupt 节点恢复,无需重跑已完成节点。"""
    flow1 = AftersaleFlow(db_path)
    flow1.start("t-5", "20260701002")
    # 新实例 = 进程重启,checkpoint 从 SQLite 恢复
    flow2 = AftersaleFlow(db_path)
    question = flow2.pending_question("t-5")
    assert question and "7 天无理由" in question
    view = flow2.resume("t-5", "确认")
    assert view["status"] == "done"
    assert "工单" in view["result"]


def test_overdue_order_goes_to_human(db_path):
    """超期订单走转人工方案。"""
    conn = ensure_db(db_path)
    conn.execute("UPDATE orders SET created_at='2026-06-01 10:00:00' "
                 "WHERE order_id='20260701002'")
    conn.commit()
    flow = AftersaleFlow(db_path)
    view = flow.start("t-6", "20260701002")
    assert "已超 7 天" in view["question"]
    view = flow.resume("t-6", "确认")
    row = flow.conn.execute(
        "SELECT issue_type FROM tickets ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == "转人工"
