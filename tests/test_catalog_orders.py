"""商品升级和会话订单列表 / Catalog migration and buyer-scoped order tests."""

from __future__ import annotations

import os
import sys
import sqlite3

import pytest

sys.path.append(os.getcwd())

from shopharness.cli import build_harness
from shopharness.llm.mock_client import MockLLM
from shopharness.core.permissions import Level
from shopharness.tools.servers import build_registry
from shopharness.data.seed import SCHEMA, PRODUCTS, ORDERS, ensure_db


def test_fifty_seed_products_and_repeat_initialization(tmp_path):
    """Verify 50 unique products and idempotent seeding at tmp_path."""
    assert len(PRODUCTS) == len({row[0] for row in PRODUCTS}) == 50
    for sku, name, category, price, min_price, stock, points in PRODUCTS:
        assert sku and name and category and points
        assert 0 < min_price <= price and stock >= 0
    path = str(tmp_path / "shop.db")
    conn = ensure_db(path)
    counts = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
              for table in ("products", "orders", "coupons", "faqs")}
    assert counts == {"products": 50, "orders": 3, "coupons": 5, "faqs": 8}
    conn.execute("UPDATE products SET price = 990, stock = 12 WHERE sku = 'YX-1001'")
    conn.execute("UPDATE orders SET amount = 950, note = 'keep me' WHERE order_id = '20260701001'")
    conn.commit()
    conn.close()
    conn = ensure_db(path)
    assert {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in counts} == counts
    assert tuple(conn.execute("SELECT price, stock FROM products WHERE sku = 'YX-1001'").fetchone()) == (990, 12)
    assert tuple(conn.execute("SELECT amount, note FROM orders WHERE order_id = '20260701001'").fetchone()) == (950, "keep me")
    conn.close()


def test_upgrade_legacy_database_preserves_unknown_order_ownership(tmp_path):
    """Upgrade a legacy schema at tmp_path while preserving business edits."""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    legacy_schema = SCHEMA.replace(",\n    buyer_id TEXT NOT NULL DEFAULT ''", "")
    conn.executescript(legacy_schema)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", PRODUCTS[:20])
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", ORDERS)
    custom_order = ("20269990001", *ORDERS[0][1:])
    conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", custom_order)
    conn.execute("UPDATE orders SET note = 'legacy note', amount = 945 WHERE order_id = '20260701001'")
    conn.commit()
    conn.close()

    conn = ensure_db(path)
    assert conn.execute("SELECT count(*) FROM products").fetchone()[0] == 50
    assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] == 4
    assert tuple(conn.execute("SELECT amount, note, buyer_id FROM orders WHERE order_id = '20260701001'").fetchone()) == (945, "legacy note", "buyer-demo")
    assert conn.execute("SELECT buyer_id FROM orders WHERE order_id = '20269990001'").fetchone()[0] == ""
    conn.execute("UPDATE orders SET buyer_id = 'buyer-other' WHERE order_id = '20260701002'")
    conn.commit()
    conn.close()
    conn = ensure_db(path)
    assert conn.execute("SELECT buyer_id FROM orders WHERE order_id = '20260701002'").fetchone()[0] == "buyer-other"
    conn.close()


def test_list_orders_is_bound_to_session_buyer(tmp_path):
    """Ensure tools built on tmp_path cannot enumerate another buyer's orders."""
    conn = ensure_db(str(tmp_path / "shop.db"))
    conn.execute("UPDATE orders SET buyer_id = 'buyer-other' WHERE order_id = '20260701002'")
    conn.commit()
    tool = build_registry(conn, buyer_id = "buyer-demo").get("list_orders")
    result = tool.execute({})
    assert result["count"] == 2
    assert [row["order_id"] for row in result["orders"]] == ["20260701003", "20260701001"]
    assert all("address" not in row and "buyer" not in row for row in result["orders"])
    assert tool.level == Level.READ
    assert tool.parameters["properties"] == {}
    assert tool.parameters["additionalProperties"] is False
    with pytest.raises(TypeError):
        tool.execute({"buyer_id": "buyer-other"})
    other = build_registry(conn, buyer_id = "buyer-other").get("list_orders").execute({})
    assert [row["order_id"] for row in other["orders"]] == ["20260701002"]
    for buyer in ("buyer-new", "anonymous", "", "x' OR 1=1 --"):
        assert build_registry(conn, buyer_id = buyer).get("list_orders").execute({}) == {"count": 0, "orders": []}
    conn.close()


@pytest.mark.parametrize("query", [
    "查询我的当前订单", "查询我当前的所有订单", "我想查询我的所有订单",
])
def test_list_order_conversations_do_not_handoff(settings, query):
    """Answer query with real tools under isolated settings without requesting an ID."""
    harness = build_harness(settings, MockLLM(), buyer_id = "buyer-demo")
    result = harness.handle(query)
    assert not result.handed_off
    assert any(event.type == "tool_call" and event.detail.startswith("list_orders(") for event in result.events)
    assert all(order[0] in result.reply for order in ORDERS)
    assert "请提供" not in result.reply
    assert harness.conn.execute("SELECT count(*) FROM tickets").fetchone()[0] == 0


def test_empty_order_list_does_not_create_ticket(settings):
    """Use settings to verify a buyer without orders receives a normal answer."""
    harness = build_harness(settings, MockLLM(), buyer_id = "buyer-new")
    result = harness.handle("查询我的全部订单")
    assert "暂无订单" in result.reply and not result.handed_off
    assert harness.conn.execute("SELECT count(*) FROM tickets").fetchone()[0] == 0


def test_new_product_can_be_found_in_mock_conversation(settings):
    """Find an added product through skill routing and tools under settings."""
    harness = build_harness(settings, MockLLM())
    result = harness.handle("露营椅")
    assert "YX-8005" in result.reply
    assert not result.handed_off


def test_exact_product_matches_rank_above_shared_characters(tmp_path):
    """Keep headphones ahead of unrelated products sharing a character in tmp_path."""
    conn = ensure_db(str(tmp_path / "shop.db"))
    result = build_registry(conn).get("search_products").execute({"keyword": "耳机"})
    assert [row["sku"] for row in result["products"][:3]] == ["YX-1002", "YX-1001", "YX-1003"]
    conn.close()
