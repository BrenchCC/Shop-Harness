"""云端向量配置、协议和缓存回归 / Offline embedding integration tests."""

from __future__ import annotations

import os
import sys
import json
import hashlib
import builtins
from pathlib import Path

import httpx
import pytest
import numpy as np
from openai import OpenAI

sys.path.append(os.getcwd())

from shopharness import cli
from shopharness.data.seed import ensure_db
from shopharness.tools.servers import build_registry
from shopharness.core.rag import create_vector_store
from shopharness.config import Settings, CloudEmbeddingSettings, load_embedding_settings
from shopharness.llm.embedding_client import EmbeddingError, OpenAIEmbedder


@pytest.fixture(autouse = True)
def isolated_config(monkeypatch, tmp_path):
    """Isolate environment values using monkeypatch and a temporary tmp_path."""
    for name in (
        "LLM_BASE_URL", "LLM_API_KEY", "MODEL", "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "EMBEDDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.chdir(tmp_path)


def cloud_settings(model: str = "embedding-model") -> CloudEmbeddingSettings:
    """Return fictional credentials for model."""
    return CloudEmbeddingSettings(
        base_url = "https://embedding.example/compatible-mode/v1",
        api_key = "embedding-secret",
        model = model,
    )


def transport(monkeypatch, handler):
    """Use monkeypatch to send embedding requests to handler without networking."""
    def factory(**kwargs):
        """Construct the SDK from kwargs using a mock transport."""
        return OpenAI(
            **kwargs,
            http_client = httpx.Client(transport = httpx.MockTransport(handler)),
        )
    monkeypatch.setattr("shopharness.llm.embedding_client.OpenAI", factory)


def success_response(request, dimensions: int = 3) -> httpx.Response:
    """Return deterministic vectors for request using dimensions columns."""
    data = []
    for i, text in enumerate(json.loads(request.content)["input"]):
        digest = hashlib.sha256(text.encode()).digest()
        data.append({"index": i, "embedding": [float(x + 1) for x in digest[:dimensions]]})
    return httpx.Response(200, json = {"data": list(reversed(data))})


def test_embedding_config_alias_and_precedence(monkeypatch):
    """Check alias compatibility, independent credentials and precedence with monkeypatch."""
    assert load_embedding_settings() is None
    Path(".env").write_text(
        "LLM_API_KEY=chat-secret\nMODEL=chat-model\n"
        "EMBEDDING_BASE_URL=https://embedding.example/v1\n"
        "EMBEDDING_API_KEY=embedding-secret\nEMBEDDDING_MODEL=legacy-model\n",
    )
    config = load_embedding_settings()
    assert config.model == "legacy-model"
    assert config.api_key.get_secret_value() == "embedding-secret"
    assert "embedding-secret" not in repr(config)
    monkeypatch.setenv("EMBEDDING_MODEL", "process-model")
    assert load_embedding_settings().model == "process-model"
    monkeypatch.delenv("EMBEDDING_MODEL")
    monkeypatch.setenv("EMBEDDDING_MODEL", "process-alias")
    assert load_embedding_settings().model == "process-alias"
    monkeypatch.setenv("EMBEDDING_MODEL", "canonical-wins")
    assert load_embedding_settings().model == "canonical-wins"
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    with pytest.raises(ValueError, match = "EMBEDDING_API_KEY"):
        load_embedding_settings()


def test_partial_embedding_config_does_not_use_chat_key(monkeypatch):
    """Do not reuse chat credentials for a partial embedding config via monkeypatch."""
    monkeypatch.setenv("LLM_API_KEY", "chat-secret")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")
    with pytest.raises(ValueError) as error:
        load_embedding_settings()
    assert "EMBEDDING_API_KEY" in str(error.value)
    assert "chat-secret" not in str(error.value)


def test_openai_batches_order_and_normalization(monkeypatch):
    """Verify protocol, batch boundaries and reordered response handling with monkeypatch."""
    requests = []

    def handler(request):
        """Capture request and return reversed vectors with known row positions."""
        assert request.url.path == "/compatible-mode/v1/embeddings"
        assert request.headers["authorization"] == "Bearer embedding-secret"
        body = json.loads(request.content)
        assert body["encoding_format"] == "float"
        assert body["model"] == "embedding-model"
        assert "dimensions" not in body
        requests.append(body["input"])
        rows = [{"index": i, "embedding": [float(i + 1), 1.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json = {"data": list(reversed(rows))})

    transport(monkeypatch, handler)
    embedder = OpenAIEmbedder(cloud_settings())
    assert embedder.embed([]).shape == (0, 0)
    texts = [f"text-{i}" for i in range(23)]
    matrix = embedder.embed(texts)
    assert [len(batch) for batch in requests] == [10, 10, 3]
    assert sum(requests, []) == texts
    assert matrix.shape == (23, 2)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis = 1), 1.0, atol = 1e-6)
    np.testing.assert_allclose(matrix[1], np.array([2, 1]) / np.sqrt(5), atol = 1e-6)
    embedder.close()


@pytest.mark.parametrize("data", [
    [],
    [{"index": 1, "embedding": [1, 2]}],
    [{"index": 0, "embedding": []}],
    [{"index": 0, "embedding": [0, 0]}],
    [{"index": 0, "embedding": ["NaN", 1]}],
    [{"index": 0, "embedding": [1, 2]}, {"index": 0, "embedding": [2, 3]}],
])
def test_invalid_embedding_responses(monkeypatch, data):
    """Reject invalid data from a monkeypatch HTTP response."""
    transport(monkeypatch, lambda request: httpx.Response(200, json = {"data": data}))
    embedder = OpenAIEmbedder(cloud_settings())
    with pytest.raises(EmbeddingError):
        embedder.embed(["query"])
    embedder.close()


def test_embedding_dimension_drift(monkeypatch):
    """Reject dimension changes across requests intercepted with monkeypatch."""
    dimensions = 3
    transport(monkeypatch, lambda request: success_response(request, dimensions))
    embedder = OpenAIEmbedder(cloud_settings())
    embedder.embed(["first"])
    dimensions = 2
    with pytest.raises(EmbeddingError, match = "维度"):
        embedder.embed(["second"])
    embedder.close()


@pytest.mark.parametrize("status", [401, 429, 500])
def test_embedding_errors_are_sanitized(monkeypatch, status):
    """Suppress response secrets for HTTP status failures using monkeypatch."""
    transport(monkeypatch, lambda request: httpx.Response(
        status, json = {"error": {"message": "embedding-secret"}},
    ))
    embedder = OpenAIEmbedder(cloud_settings())
    with pytest.raises(EmbeddingError) as error:
        embedder.embed(["hello"])
    assert str(status) in str(error.value)
    assert "embedding-secret" not in str(error.value)
    embedder.close()


def test_cache_reuse_update_model_switch_and_failure(monkeypatch, tmp_path):
    """Exercise persistent caches, source changes and failed replacement in tmp_path."""
    requests = []
    failing = False

    def handler(request):
        """Record request inputs and optionally fail a cache rebuild."""
        requests.append(json.loads(request.content)["input"])
        if failing:
            return httpx.Response(500, json = {"error": {"message": "embedding-secret"}})
        return success_response(request)

    transport(monkeypatch, handler)
    conn = ensure_db(str(tmp_path / "shop.db"))
    store = create_vector_store(conn, "missing-local-model", cloud_settings())
    assert store is not None
    count = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert sum(map(len, requests)) == count
    store.embedder.close()
    requests.clear()
    reused = create_vector_store(conn, "missing-local-model", cloud_settings())
    assert reused is not None and requests == []
    reused.embedder.close()
    conn.execute("UPDATE products SET selling_points = selling_points || ' changed' WHERE sku = 'YX-1001'")
    conn.commit()
    updated = create_vector_store(conn, "missing-local-model", cloud_settings())
    assert updated is not None and sum(map(len, requests)) == 1
    updated.embedder.close()
    requests.clear()
    switched = create_vector_store(conn, "missing-local-model", cloud_settings("different-model"))
    assert switched is not None and sum(map(len, requests)) == count
    before = [tuple(row) for row in conn.execute("SELECT * FROM embeddings")]
    metadata = tuple(conn.execute("SELECT * FROM embedding_metadata").fetchone())
    failing = True
    assert create_vector_store(conn, "missing-local-model", cloud_settings("failing-model")) is None
    assert [tuple(row) for row in conn.execute("SELECT * FROM embeddings")] == before
    assert tuple(conn.execute("SELECT * FROM embedding_metadata").fetchone()) == metadata
    # 查询失败不影响业务关键词检索 / Query outages still allow keyword results.
    registry = build_registry(conn, switched)
    assert registry.get("search_products").execute({"keyword": "耳机"})["count"] > 0
    assert registry.get("search_faq").execute({"query": "发票"})["count"] > 0
    switched.embedder.close()
    conn.close()


def test_legacy_vectors_are_rebuilt(monkeypatch, tmp_path):
    """Prevent mixing an unversioned local cache in tmp_path with cloud vectors."""
    transport(monkeypatch, success_response)
    conn = ensure_db(str(tmp_path / "shop.db"))
    conn.execute("CREATE TABLE embeddings (doc_type TEXT, doc_id TEXT, text TEXT, vector TEXT, PRIMARY KEY (doc_type, doc_id))")
    conn.execute("INSERT INTO embeddings VALUES ('product', 'obsolete', 'old', '[1,0]')")
    conn.commit()
    store = create_vector_store(conn, "missing-local-model", cloud_settings())
    assert store is not None
    assert conn.execute("SELECT count(*) FROM embeddings WHERE doc_id = 'obsolete'").fetchone()[0] == 0
    assert store.dimensions == 3
    assert store.search("query", "product")
    store.embedder.close()
    conn.close()


def test_mock_cli_ignores_embedding_configuration(monkeypatch, tmp_path):
    """Ensure --mock skips private embedding config using monkeypatch and tmp_path."""
    monkeypatch.setenv("EMBEDDING_MODEL", "incomplete-must-be-ignored")
    monkeypatch.setattr(sys, "argv", ["shopharness", "--mock"])
    monkeypatch.setattr("builtins.input", lambda prompt: "退出")
    settings = Settings(
        db_path = str(tmp_path / "shop.db"),
        trace_dir = str(tmp_path / "traces"),
        rag_enabled = False,
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    assert cli.main() == 0
    assert settings.cloud_embedding is None


@pytest.mark.parametrize("argv", [[], ["--endpoint", "http://localhost:8000/v1"]])
def test_cli_loads_cloud_vectors_without_local_inference(monkeypatch, tmp_path, argv):
    """Wire cloud vectors for argv using monkeypatch and isolated tmp_path storage."""
    config = cloud_settings()
    monkeypatch.setenv("EMBEDDING_BASE_URL", config.base_url)
    monkeypatch.setenv("EMBEDDING_API_KEY", config.api_key.get_secret_value())
    monkeypatch.setenv("EMBEDDDING_MODEL", config.model)
    monkeypatch.setattr(sys, "argv", ["shopharness", *argv])
    monkeypatch.setattr(builtins, "input", lambda prompt: "退出")
    original_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        """Record forbidden module name imports and forward other args and kwargs."""
        if name.split(".")[0] in {"vllm", "torch", "transformers"}:
            attempted.append(name)
            raise AssertionError("Cloud embedding must not load local inference")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    transport(monkeypatch, success_response)
    settings = Settings(
        db_path = str(tmp_path / "shop.db"),
        trace_dir = str(tmp_path / "traces"),
        skills_dir = str(Path(__file__).resolve().parents[1] / "skills"),
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    assert cli.main() == 0
    assert settings.cloud_embedding.model == config.model
    conn = ensure_db(settings.db_path)
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] > 0
    assert attempted == []
    conn.close()
