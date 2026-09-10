"""FastMCP server discovery, execution, and safety tests."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from shopharness.mcp_server import create_mcp_server


def test_mcp_lists_and_calls_business_tools(tmp_path):
    """Verify MCP discovery and a read tool against an isolated database."""
    db_path = str(tmp_path / "shop.db")
    server = create_mcp_server(db_path, "buyer-demo")

    async def exercise() -> None:
        """Run an in-memory MCP client scenario."""
        async with Client(server) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools}
            assert names == {
                "add_order_note",
                "adjust_price",
                "calc_discount",
                "create_ticket",
                "get_logistics",
                "get_order",
                "get_product_detail",
                "list_orders",
                "search_faq",
                "search_products",
            }
            result = await client.call_tool(
                "search_products",
                {"keyword": "耳机"},
            )
            assert result.data["count"] == 5
            assert {
                product["sku"] for product in result.data["products"][:3]
            } == {"YX-1001", "YX-1002", "YX-1003"}

    asyncio.run(exercise())


def test_mcp_price_change_requires_confirmation_and_guardrail(tmp_path):
    """Require confirmation and retain the minimum-price guardrail over MCP."""
    db_path = str(tmp_path / "shop.db")
    server = create_mcp_server(db_path, "buyer-demo")

    async def exercise() -> None:
        """Run dangerous MCP calls against an isolated database."""
        async with Client(server) as client:
            args = {
                "order_id": "20260701001",
                "new_price": 900,
                "reason": "测试",
            }
            with pytest.raises(Exception, match = "confirmed=true"):
                await client.call_tool("adjust_price", args)
            with pytest.raises(Exception, match = "最低限价"):
                await client.call_tool(
                    "adjust_price",
                    {**args, "new_price": 800, "confirmed": True},
                )
            result = await client.call_tool(
                "adjust_price",
                {**args, "confirmed": True},
            )
            assert result.data["new_amount"] == 900

    asyncio.run(exercise())
    conn = sqlite3.connect(db_path)
    assert conn.execute(
        "SELECT count(*) FROM audit WHERE tool = 'adjust_price'"
    ).fetchone()[0] == 1
    conn.close()
