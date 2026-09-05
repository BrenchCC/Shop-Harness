"""RAG:向量语义检索、RRF 混合、FAQ 知识库、优雅降级。

需要 models/bge-small-zh-v1.5(scripts 下载);模型加载只做一次。
"""

from __future__ import annotations

import pytest

from shopharness.core.rag import (VectorStore, create_vector_store, rrf_fuse)
from shopharness.data.seed import ensure_db
from shopharness.tools.servers import build_registry

MODEL_PATH = "models/bge-small-zh-v1.5"


@pytest.fixture(scope="session")
def shared_db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rag") / "shop.db")
    return ensure_db(path)


@pytest.fixture(scope="session")
def store(shared_db):
    return VectorStore(shared_db, MODEL_PATH)


def test_vector_search_semantic_product(store):
    """口语化查询(无关键词重叠)也能命中语义相关商品。"""
    results = store.search("睡觉脖子疼,想买保护颈椎的", doc_type="product",
                           top_k=3)
    ids = [doc_id for doc_id, _ in results]
    assert "YX-3001" in ids  # 云朵记忆棉枕头


def test_vector_search_faq(store):
    results = store.search("买错了能不能退", doc_type="faq", top_k=2)
    assert results
    assert results[0][1] > 0.4


def test_rrf_fuse():
    fused = rrf_fuse(["a", "b", "c"], ["b", "d", "a"])
    assert fused[0] in ("a", "b")  # 两路都靠前的排最前
    assert set(fused) == {"a", "b", "c", "d"}


def test_hybrid_search_products(shared_db, store):
    registry = build_registry(shared_db, store)
    result = registry.get("search_products").execute(
        {"keyword": "保护颈椎助睡眠"})
    skus = [p["sku"] for p in result["products"]]
    assert "YX-3001" in skus


def test_search_faq_tool(shared_db, store):
    registry = build_registry(shared_db, store)
    result = registry.get("search_faq").execute({"query": "退货要几天"})
    assert result["count"] >= 1
    assert "7 天" in result["faqs"][0]["answer"]


def test_graceful_degradation_when_model_missing(tmp_path):
    conn = ensure_db(str(tmp_path / "shop.db"))
    assert create_vector_store(conn, "models/not-exist") is None
    # 降级后关键词检索仍可用
    registry = build_registry(conn, None)
    result = registry.get("search_products").execute({"keyword": "耳机"})
    assert result["count"] >= 1


def test_keyword_fallback_search_faq(tmp_path):
    conn = ensure_db(str(tmp_path / "shop.db"))
    registry = build_registry(conn, None)
    result = registry.get("search_faq").execute({"query": "发票"})
    assert result["count"] >= 1
