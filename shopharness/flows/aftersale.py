"""退换货售后长流程(LangGraph + SqliteSaver checkpoint)。

流程:核实订单 → 售后期限判断 → 方案确认(interrupt,等买家输入)→ 执行 → 结束
checkpoint 落 SQLite:进程重启后可从 interrupt 节点恢复(对应"跨天售后"场景)。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

RETURN_WINDOW_DAYS = 7


class FlowState(TypedDict, total=False):
    order_id: str
    product_name: str
    order_status: str
    within_window: bool
    plan: str
    approved: bool
    result: str


def _days_since(date_str: str) -> int:
    created = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return (datetime.now() - created).days


def build_graph(conn: sqlite3.Connection, saver: SqliteSaver):

    def verify_order(state: FlowState) -> dict[str, Any]:
        row = conn.execute(
            "SELECT o.*, p.name AS product_name FROM orders o "
            "JOIN products p ON p.sku = o.sku WHERE o.order_id = ?",
            (state["order_id"],)).fetchone()
        if not row:
            return {"result": f"订单 {state['order_id']} 不存在,流程终止。"}
        within = _days_since(row["created_at"]) <= RETURN_WINDOW_DAYS
        plan = (f"已核实订单 {state['order_id']}({row['product_name']},"
                f"金额 {row['amount']:.0f} 元,状态:{row['status']})。")
        if within:
            plan += ("在 7 天无理由退换期内,建议方案:优先换货,"
                     "买家坚持可退货退款(原路退回,1-3 个工作日)。"
                     "请买家确认是否按此方案处理。")
        else:
            plan += ("已超 7 天无理由期,建议方案:转人工审核特殊售后。"
                     "请买家确认是否转人工。")
        return {"product_name": row["product_name"],
                "order_status": row["status"], "within_window": within,
                "plan": plan}

    def ask_buyer(state: FlowState) -> dict[str, Any]:
        """interrupt 点:流程挂起,等买家对方案的答复(可跨进程恢复)。"""
        answer = interrupt({"question": state["plan"]})
        approved = any(kw in str(answer)
                       for kw in ("确认", "同意", "好的", "可以", "行"))
        return {"approved": approved}

    def execute(state: FlowState) -> dict[str, Any]:
        if not state.get("approved"):
            return {"result": "买家未确认方案,流程结束,保持原订单状态。"}
        if state.get("within_window"):
            issue, summary = "售后", (
                f"买家确认退换方案:订单 {state['order_id']}"
                f"({state.get('product_name')}),优先换货处理。")
        else:
            issue, summary = "转人工", (
                f"超期售后审核:订单 {state['order_id']},买家已确认转人工。")
        cur = conn.execute(
            "INSERT INTO tickets(issue_type, summary) VALUES (?,?)",
            (issue, summary))
        conn.commit()
        return {"result": f"已执行:创建工单 #{cur.lastrowid}({issue})。{summary}"}

    builder = StateGraph(FlowState)
    builder.add_node("verify_order", verify_order)
    builder.add_node("ask_buyer", ask_buyer)
    builder.add_node("execute", execute)
    builder.add_edge(START, "verify_order")
    # 订单不存在时 verify_order 已写入 result,直接终止
    builder.add_conditional_edges(
        "verify_order",
        lambda state: "end" if state.get("result") else "ask",
        {"end": END, "ask": "ask_buyer"})
    builder.add_edge("ask_buyer", "execute")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=saver)


class AftersaleFlow:
    """对外门面:start/resume/state,checkpoint 与业务库分文件。"""

    def __init__(self, db_path: str, checkpoint_path: str | None = None):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        cp_path = checkpoint_path or db_path.replace(".db", "-checkpoints.db")
        saver = SqliteSaver(sqlite3.connect(cp_path, check_same_thread=False))
        self.graph = build_graph(self.conn, saver)

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(self, thread_id: str, order_id: str) -> dict[str, Any]:
        result = self.graph.invoke({"order_id": order_id},
                                   self._config(thread_id))
        return self._view(thread_id, result)

    def resume(self, thread_id: str, buyer_input: str) -> dict[str, Any]:
        result = self.graph.invoke(Command(resume=buyer_input),
                                   self._config(thread_id))
        return self._view(thread_id, result)

    def _view(self, thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
        interrupts = result.get("__interrupt__")
        if interrupts:
            question = interrupts[0].value.get("question", "")
            return {"status": "waiting_buyer", "question": question}
        return {"status": "done", "result": result.get("result", "")}

    def pending_question(self, thread_id: str) -> str | None:
        """进程重启后查询某会话是否停在等待买家节点。"""
        state = self.graph.get_state(self._config(thread_id))
        for task in state.tasks or []:
            if task.interrupts:
                return task.interrupts[0].value.get("question")
        return None
