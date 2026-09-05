"""8 个业务工具的 SQLite 实现 + 护栏/审计 hook 工厂。

约定:
- 工具返回 dict;业务性失败返回 {"error": "..."}(软失败,回注给模型)
- 参数校验失败抛 ToolError(硬失败,计入熔断)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..core.hooks import HookResult
from ..core.permissions import Level
from ..core.rag import VectorStore, rrf_fuse
from .registry import Tool, ToolError, ToolRegistry


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ---------------------------------------------------------------- 工具实现

def make_tools(conn: sqlite3.Connection,
               vector_store: "VectorStore | None" = None) -> list[Tool]:

    def search_products(keyword: str, category: str | None = None) -> dict[str, Any]:
        if not keyword or not keyword.strip():
            raise ToolError("keyword 不能为空")
        terms = [t for t in keyword.split() if t]
        rows = conn.execute("SELECT * FROM products").fetchall()

        def score_rows(cat: str | None) -> list[tuple[int, sqlite3.Row]]:
            scored: list[tuple[int, sqlite3.Row]] = []
            for row in rows:
                if cat and cat not in row["category"]:
                    continue
                haystack = (f"{row['sku']} {row['name']} {row['category']} "
                            f"{row['selling_points']}")
                score = sum(haystack.count(t) for t in terms)
                # 单字关键词也做包含匹配,兼容中文搜索习惯
                if score == 0 and any(t in haystack for t in keyword):
                    score = 1
                if score > 0:
                    scored.append((score, row))
            scored.sort(key=lambda x: (-x[0], x[1]["price"]))
            return scored

        note = None
        if vector_store is not None:
            # RAG 混合检索:关键词(精确/SKU 友好) + 向量(语义/口语化友好),RRF 融合
            kw_ids = [r["sku"] for _, r in score_rows(category)]
            vec_ids = [doc_id for doc_id, sim in
                       vector_store.search(keyword, doc_type="product", top_k=8)
                       if sim > 0.4]
            fused = rrf_fuse(kw_ids, vec_ids)
            by_sku = {r["sku"]: r for r in rows}
            ordered = [by_sku[s] for s in fused if s in by_sku]
            if not ordered and category:
                ordered = [by_sku[s] for s in
                           rrf_fuse([r["sku"] for _, r in score_rows(None)],
                                    vec_ids) if s in by_sku]
                note = f"类目「{category}」无匹配,已按全店范围检索"
            items = [{
                "sku": r["sku"], "name": r["name"], "price": r["price"],
                "stock": r["stock"],
                "selling_points": r["selling_points"][:60],
            } for r in ordered[:5]]
        else:
            scored = score_rows(category)
            if not scored and category:
                scored = score_rows(None)
                note = f"类目「{category}」无匹配,已按全店范围检索"
            items = [{
                "sku": r["sku"], "name": r["name"], "price": r["price"],
                "stock": r["stock"],
                "selling_points": r["selling_points"][:60],
            } for _, r in scored[:5]]
        if not items:
            return {"count": 0, "products": [],
                    "hint": "未找到匹配商品,可换关键词重试"}
        result: dict[str, Any] = {"count": len(items), "products": items}
        if note:
            result["note"] = note
        return result

    def search_faq(query: str) -> dict[str, Any]:
        if not query or not query.strip():
            raise ToolError("query 不能为空")
        hits: list[dict[str, Any]] = []
        if vector_store is not None:
            for doc_id, sim in vector_store.search(query, doc_type="faq",
                                                   top_k=2):
                row = conn.execute(
                    "SELECT * FROM faqs WHERE id = ?", (doc_id,)).fetchone()
                if row and sim > 0.4:
                    hits.append({"question": row["question"],
                                 "answer": row["answer"],
                                 "similarity": round(sim, 3)})
        else:
            for row in conn.execute("SELECT * FROM faqs").fetchall():
                if any(t in row["question"] + row["answer"]
                       for t in query.split()):
                    hits.append({"question": row["question"],
                                 "answer": row["answer"]})
                if len(hits) >= 2:
                    break
        return {"count": len(hits), "faqs": hits} if hits else \
            {"count": 0, "faqs": [], "hint": "知识库未覆盖该问题,可转人工"}

    def get_product_detail(sku: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
        if not row:
            return {"error": f"商品 {sku} 不存在"}
        return _row_to_dict(row)

    def get_order(order_id: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT o.*, p.name AS product_name FROM orders o "
            "JOIN products p ON p.sku = o.sku WHERE o.order_id = ?",
            (order_id,)).fetchone()
        if not row:
            return {"error": f"订单 {order_id} 不存在,请核对订单号"}
        return _row_to_dict(row)

    def get_logistics(order_id: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM logistics WHERE order_id = ?", (order_id,)).fetchone()
        if not row:
            return {"error": f"订单 {order_id} 暂无物流信息(可能未发货)"}
        return {"order_id": order_id, "status": row["status"],
                "trace": json.loads(row["trace"])}

    def calc_discount(sku: str, quantity: int = 1) -> dict[str, Any]:
        if quantity < 1:
            raise ToolError("quantity 必须 >= 1")
        row = conn.execute(
            "SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
        if not row:
            return {"error": f"商品 {sku} 不存在"}
        total = row["price"] * quantity
        best = conn.execute(
            "SELECT * FROM coupons WHERE sku = ? AND threshold <= ? "
            "ORDER BY discount DESC LIMIT 1", (sku, total)).fetchone()
        result: dict[str, Any] = {
            "sku": sku, "name": row["name"], "unit_price": row["price"],
            "quantity": quantity, "original_total": round(total, 2),
        }
        if best:
            result["coupon"] = f"满{best['threshold']:.0f}减{best['discount']:.0f}"
            result["final_total"] = round(total - best["discount"], 2)
        else:
            result["coupon"] = None
            result["final_total"] = round(total, 2)
        return result

    def add_order_note(order_id: str, note: str) -> dict[str, Any]:
        cur = conn.execute(
            "UPDATE orders SET note = ? WHERE order_id = ?", (note, order_id))
        conn.commit()
        if cur.rowcount == 0:
            return {"error": f"订单 {order_id} 不存在"}
        return {"ok": True, "order_id": order_id, "note": note}

    def adjust_price(order_id: str, new_price: float,
                     reason: str) -> dict[str, Any]:
        if new_price <= 0:
            raise ToolError("new_price 必须为正数")
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if not row:
            return {"error": f"订单 {order_id} 不存在"}
        conn.execute("UPDATE orders SET amount = ? WHERE order_id = ?",
                     (new_price, order_id))
        conn.commit()
        return {"ok": True, "order_id": order_id, "old_amount": row["amount"],
                "new_amount": new_price, "reason": reason}

    def create_ticket(issue_type: str, summary: str) -> dict[str, Any]:
        cur = conn.execute(
            "INSERT INTO tickets(issue_type, summary) VALUES (?,?)",
            (issue_type, summary))
        conn.commit()
        return {"ok": True, "ticket_id": cur.lastrowid,
                "issue_type": issue_type}

    obj = {"type": "object"}
    return [
        Tool("search_products", "按关键词检索在售商品,返回 top5(名称/价格/库存/卖点)",
             {**obj, "properties": {
                 "keyword": {"type": "string", "description": "搜索关键词"},
                 "category": {"type": "string", "description": "可选,类目过滤"}},
              "required": ["keyword"]},
             Level.READ, search_products),
        Tool("search_faq", "检索店铺知识库(退换政策/发货时效/发票/保修等),"
             "回答规则类问题前必查",
             {**obj, "properties": {
                 "query": {"type": "string", "description": "问题描述"}},
              "required": ["query"]},
             Level.READ, search_faq),
        Tool("get_product_detail", "查询单个商品完整详情(售价/最低限价/库存/卖点)",
             {**obj, "properties": {
                 "sku": {"type": "string", "description": "商品 SKU,如 YX-1001"}},
              "required": ["sku"]},
             Level.READ, get_product_detail),
        Tool("get_order", "查询订单状态、金额、商品与收货信息",
             {**obj, "properties": {
                 "order_id": {"type": "string", "description": "订单号"}},
              "required": ["order_id"]},
             Level.READ, get_order),
        Tool("get_logistics", "查询订单物流状态与轨迹",
             {**obj, "properties": {
                 "order_id": {"type": "string"}},
              "required": ["order_id"]},
             Level.READ, get_logistics),
        Tool("calc_discount", "计算商品按当前优惠规则的到手价",
             {**obj, "properties": {
                 "sku": {"type": "string"},
                 "quantity": {"type": "integer", "default": 1}},
              "required": ["sku"]},
             Level.READ, calc_discount),
        Tool("add_order_note", "给订单添加客服备注(可逆写操作)",
             {**obj, "properties": {
                 "order_id": {"type": "string"},
                 "note": {"type": "string"}},
              "required": ["order_id", "note"]},
             Level.WRITE, add_order_note),
        Tool("adjust_price",
             "修改订单金额(危险操作!须先获买家明确确认,且不得低于最低限价)",
             {**obj, "properties": {
                 "order_id": {"type": "string"},
                 "new_price": {"type": "number"},
                 "reason": {"type": "string", "description": "改价原因"}},
              "required": ["order_id", "new_price", "reason"]},
             Level.DANGEROUS, adjust_price),
        Tool("create_ticket", "创建工单/转人工单",
             {**obj, "properties": {
                 "issue_type": {"type": "string",
                                "description": "如 售后/投诉/转人工"},
                 "summary": {"type": "string"}},
              "required": ["issue_type", "summary"]},
             Level.WRITE, create_ticket),
    ]


def build_registry(conn: sqlite3.Connection,
                   vector_store: VectorStore | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in make_tools(conn, vector_store):
        registry.register(tool)
    return registry


# ---------------------------------------------------------------- 内置 hook

def make_price_guardrail(conn: sqlite3.Connection):
    """pre hook:改价不得低于商品最低限价(价格篡改检测)。"""

    def guard(tool_name: str, args: dict[str, Any]) -> HookResult:
        if tool_name != "adjust_price":
            return HookResult()
        row = conn.execute(
            "SELECT p.min_price, p.name FROM orders o "
            "JOIN products p ON p.sku = o.sku WHERE o.order_id = ?",
            (args.get("order_id"),)).fetchone()
        if not row:
            return HookResult()  # 订单不存在由工具自身报错
        if args.get("new_price", 0) < row["min_price"]:
            return HookResult.deny(
                f"改价被拒绝:{row['name']} 最低限价 {row['min_price']:.0f} 元,"
                f"申请价 {args['new_price']:.0f} 元低于红线")
        return HookResult()

    return guard


def make_audit_hook(conn: sqlite3.Connection):
    """post hook:WRITE/DANGEROUS 工具调用落审计表。"""

    def audit(tool_name: str, args: dict[str, Any],
              result: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO audit(tool, args, result) VALUES (?,?,?)",
            (tool_name, json.dumps(args, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False)))
        conn.commit()

    return audit
