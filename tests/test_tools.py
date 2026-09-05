"""工具层 CRUD 与 schema 测试。"""

from __future__ import annotations

import pytest

from shopharness.tools.registry import ToolError
from shopharness.tools.servers import build_registry


@pytest.fixture()
def registry(settings):
    from shopharness.data.seed import ensure_db
    conn = ensure_db(settings.db_path)
    return build_registry(conn)


def test_search_products_keyword(registry):
    result = registry.get("search_products").execute({"keyword": "耳机"})
    assert result["count"] >= 3
    skus = [p["sku"] for p in result["products"]]
    assert "YX-1001" in skus


def test_search_products_empty_keyword_raises(registry):
    with pytest.raises(ToolError):
        registry.get("search_products").execute({"keyword": "  "})


def test_get_order_found_and_missing(registry):
    ok = registry.get("get_order").execute({"order_id": "20260701001"})
    assert ok["status"] == "待发货"
    assert ok["product_name"] == "音弦无线降噪耳机 Pro"
    missing = registry.get("get_order").execute({"order_id": "20000000000"})
    assert "error" in missing


def test_get_logistics(registry):
    ok = registry.get("get_logistics").execute({"order_id": "20260701002"})
    assert ok["status"] == "运输中"
    assert len(ok["trace"]) == 3
    none = registry.get("get_logistics").execute({"order_id": "20260701001"})
    assert "error" in none


def test_calc_discount_with_coupon(registry):
    result = registry.get("calc_discount").execute(
        {"sku": "YX-1001", "quantity": 1})
    assert result["final_total"] == 899.0
    assert result["coupon"] == "满999减100"


def test_calc_discount_no_coupon_below_threshold(registry):
    result = registry.get("calc_discount").execute(
        {"sku": "YX-6001", "quantity": 1})
    # 单价 59.9 ≥ 50 门槛,有券
    assert result["final_total"] == 54.9


def test_adjust_price_and_note(registry, settings):
    from shopharness.data.seed import ensure_db
    conn = ensure_db(settings.db_path)
    result = registry.get("adjust_price").execute(
        {"order_id": "20260701001", "new_price": 950.0, "reason": "test"})
    assert result["ok"] and result["new_amount"] == 950.0
    row = conn.execute(
        "SELECT amount FROM orders WHERE order_id='20260701001'").fetchone()
    assert row[0] == 950.0
    note = registry.get("add_order_note").execute(
        {"order_id": "20260701001", "note": "买家已确认"})
    assert note["ok"]


def test_create_ticket(registry):
    result = registry.get("create_ticket").execute(
        {"issue_type": "转人工", "summary": "test"})
    assert result["ok"] and result["ticket_id"] >= 1


def test_tool_schema_format(registry):
    schemas = registry.schemas()
    assert len(schemas) == 9
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]


def test_tool_whitelist_filtering(registry):
    schemas = registry.schemas({"get_order", "create_ticket"})
    names = {s["function"]["name"] for s in schemas}
    assert names == {"get_order", "create_ticket"}
