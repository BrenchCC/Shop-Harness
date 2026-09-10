"""Local FastMCP server exposing ShopHarness business tools."""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError as MCPToolError

sys.path.append(os.getcwd())

from shopharness.core.permissions import Level
from shopharness.data.seed import ensure_db
from shopharness.tools.registry import ToolError
from shopharness.tools.servers import (
    build_registry,
    make_audit_hook,
    make_price_guardrail,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "shopharness/data/shop.db"


class MCPToolExecutor:
    """Execute registry tools with isolated connections and safety hooks."""

    def __init__(self, db_path: str, buyer_id: str):
        """Initialize the executor.

        Args:
            db_path: Path to the ShopHarness SQLite database.
            buyer_id: Trusted buyer identity bound by the MCP process.
        """
        self.db_path = db_path
        self.buyer_id = buyer_id

    def call(
        self,
        name: str,
        args: dict[str, Any],
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Execute one registered tool.

        Args:
            name: Registered ShopHarness tool name.
            args: Validated tool arguments received from FastMCP.
            confirmed: Whether the caller explicitly confirmed a dangerous action.
        """
        conn = ensure_db(self.db_path)
        try:
            tool = build_registry(conn, buyer_id = self.buyer_id).get(name)
            if tool is None:
                raise MCPToolError(f"工具 {name} 不存在")
            if tool.level is Level.DANGEROUS:
                if not confirmed:
                    raise MCPToolError("危险操作必须设置 confirmed=true 明确确认")
                guard = make_price_guardrail(conn)(name, args)
                if not guard.allow:
                    raise MCPToolError(guard.reason or "危险操作被护栏拒绝")
            try:
                result = tool.execute(args)
            except ToolError as exc:
                raise MCPToolError(str(exc)) from None
            if "error" not in result and tool.level in (Level.WRITE, Level.DANGEROUS):
                make_audit_hook(conn)(name, args, result)
            return result
        finally:
            conn.close()


def create_mcp_server(db_path: str, buyer_id: str) -> FastMCP:
    """Create the local ShopHarness MCP server.

    Args:
        db_path: Path to the ShopHarness SQLite database.
        buyer_id: Trusted buyer identity bound to order-list queries.
    """
    executor = MCPToolExecutor(db_path, buyer_id)
    server = FastMCP(name = "ShopHarness")

    @server.tool(description = "按关键词检索在售商品，返回最多 5 个商品")
    def search_products(keyword: str, category: str | None = None) -> dict[str, Any]:
        """Search products.

        Args:
            keyword: Product search keywords.
            category: Optional product category filter.
        """
        return executor.call(
            "search_products",
            {"keyword": keyword, "category": category},
        )

    @server.tool(description = "检索店铺 FAQ 知识库")
    def search_faq(query: str) -> dict[str, Any]:
        """Search FAQs.

        Args:
            query: Customer question to search for.
        """
        return executor.call("search_faq", {"query": query})

    @server.tool(description = "按 SKU 查询商品完整详情")
    def get_product_detail(sku: str) -> dict[str, Any]:
        """Get product details.

        Args:
            sku: Product SKU.
        """
        return executor.call("get_product_detail", {"sku": sku})

    @server.tool(description = "查询当前 MCP 买家身份绑定的全部订单")
    def list_orders() -> dict[str, Any]:
        """List orders for the buyer identity configured on the MCP process."""
        return executor.call("list_orders", {})

    @server.tool(description = "按订单号查询订单详情")
    def get_order(order_id: str) -> dict[str, Any]:
        """Get an order.

        Args:
            order_id: Order identifier.
        """
        return executor.call("get_order", {"order_id": order_id})

    @server.tool(description = "按订单号查询物流状态和轨迹")
    def get_logistics(order_id: str) -> dict[str, Any]:
        """Get logistics information.

        Args:
            order_id: Order identifier.
        """
        return executor.call("get_logistics", {"order_id": order_id})

    @server.tool(description = "计算指定商品和数量的优惠后价格")
    def calc_discount(sku: str, quantity: int = 1) -> dict[str, Any]:
        """Calculate a product discount.

        Args:
            sku: Product SKU.
            quantity: Positive purchase quantity.
        """
        return executor.call("calc_discount", {"sku": sku, "quantity": quantity})

    @server.tool(description = "给订单添加客服备注；该写操作会记录审计日志")
    def add_order_note(order_id: str, note: str) -> dict[str, Any]:
        """Add an order note.

        Args:
            order_id: Order identifier.
            note: Customer-service note.
        """
        return executor.call("add_order_note", {"order_id": order_id, "note": note})

    @server.tool(description = "修改订单金额；必须明确确认且不能低于商品最低限价")
    def adjust_price(
        order_id: str,
        new_price: float,
        reason: str,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Adjust an order price after explicit confirmation.

        Args:
            order_id: Order identifier.
            new_price: Requested new order amount.
            reason: Reason for changing the price.
            confirmed: Explicit confirmation for this dangerous operation.
        """
        return executor.call(
            "adjust_price",
            {"order_id": order_id, "new_price": new_price, "reason": reason},
            confirmed = confirmed,
        )

    @server.tool(description = "创建客服工单；该写操作会记录审计日志")
    def create_ticket(issue_type: str, summary: str) -> dict[str, Any]:
        """Create a customer-service ticket.

        Args:
            issue_type: Ticket category such as after-sales or complaint.
            summary: Concise issue summary.
        """
        return executor.call(
            "create_ticket",
            {"issue_type": issue_type, "summary": summary},
        )

    return server


def parse_args() -> argparse.Namespace:
    """Parse local MCP server arguments."""
    parser = argparse.ArgumentParser(description = "ShopHarness 本地 MCP Server")
    parser.add_argument(
        "--transport",
        choices = ("stdio", "http"),
        default = "stdio",
        help = "本地客户端默认使用 stdio；调试可使用 http",
    )
    parser.add_argument("--host", default = "127.0.0.1")
    parser.add_argument("--port", type = int, default = 8001)
    parser.add_argument(
        "--db",
        default = os.getenv("SHOPHARNESS_DB_PATH", str(DEFAULT_DB_PATH)),
    )
    parser.add_argument(
        "--buyer-id",
        default = os.getenv("SHOPHARNESS_BUYER_ID", "anonymous"),
    )
    return parser.parse_args()


mcp = create_mcp_server(
    os.getenv("SHOPHARNESS_DB_PATH", str(DEFAULT_DB_PATH)),
    os.getenv("SHOPHARNESS_BUYER_ID", "anonymous"),
)


if __name__ == "__main__":
    args = parse_args()
    server = create_mcp_server(args.db, args.buyer_id)
    if args.transport == "stdio":
        server.run()
    else:
        server.run(transport = "http", host = args.host, port = args.port)
