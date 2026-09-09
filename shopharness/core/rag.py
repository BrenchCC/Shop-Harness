"""RAG 检索增强:云端或本地向量 + 关键词检索,RRF 混合排序。

设计要点:
- 只用 transformers 直读 bge 模型(mean pooling + L2 归一化),不引 sentence-transformers
- 向量存 SQLite embeddings 表,按模型身份与维度校验,失败时降级为关键词检索
- CPU 推理(bge-small 单次毫秒级),不占用 vLLM 显存
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np

sys.path.append(os.getcwd())

from ..config import CloudEmbeddingSettings
from ..llm.embedding_client import EmbeddingError, OpenAIEmbedder, normalize_vectors

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    doc_type TEXT NOT NULL,      -- product / faq
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    vector TEXT NOT NULL,        -- JSON 数组
    PRIMARY KEY (doc_type, doc_id)
);
CREATE TABLE IF NOT EXISTS embedding_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    index_key TEXT NOT NULL,
    dimensions INTEGER NOT NULL
);
"""
# FAQ 表结构与种子数据见 data/seed.py(业务数据统一入口)


class Embedder:
    """bge-small-zh 文本向量(mean pooling + 归一化),CPU 推理。"""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        identity = "local-bge-l2-v1:" + str(Path(model_path).resolve())
        self.index_key = hashlib.sha256(identity.encode()).hexdigest()
        # 可选本地检索不自动下载 / Optional local retrieval never downloads weights.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only = True)
        self.model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            batch = self.tokenizer(texts, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")
            out = self.model(**batch).last_hidden_state  # (B, T, H)
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = self.torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.numpy()


class VectorStore:
    """商品 + FAQ 的向量索引,SQLite 持久化,增量构建。"""

    def __init__(
        self,
        conn: sqlite3.Connection,
        model_path: str,
        embedder: OpenAIEmbedder | None = None,
    ):
        """Use conn for storage and embedder when supplied, otherwise load model_path."""
        self.conn = conn
        self.conn.executescript(SCHEMA)
        self.embedder = embedder if embedder is not None else Embedder(model_path)
        self.index_key = self.embedder.index_key
        self._ensure_index()

    def _doc_texts(self) -> list[tuple[str, str, str]]:
        docs = []
        for row in self.conn.execute("SELECT * FROM products").fetchall():
            text = (f"{row['sku']} {row['name']} {row['category']} "
                    f"{row['selling_points']}")
            docs.append(("product", row["sku"], text))
        for row in self.conn.execute("SELECT * FROM faqs").fetchall():
            docs.append(("faq", str(row["id"]),
                         f"{row['question']} {row['answer']}"))
        return docs

    def _ensure_index(self) -> None:
        """Refresh changed documents and replace incompatible caches atomically."""
        metadata = self.conn.execute(
            "SELECT index_key, dimensions FROM embedding_metadata WHERE id = 1",
        ).fetchone()
        compatible = metadata is not None and metadata[0] == self.index_key
        existing = {(r[0], r[1]): r[2] for r in self.conn.execute(
            "SELECT doc_type, doc_id, text FROM embeddings",
        )} if compatible else {}
        docs = self._doc_texts()
        pending = [(t, i, text) for t, i, text in docs if existing.get((t, i)) != text]
        self.dimensions = (metadata[1] or None) if compatible else None
        vectors = []
        if pending:
            # 全部成功后再替换缓存 / Do not overwrite the cache on an API failure.
            vectors = normalize_vectors(
                self.embedder.embed([text for _, _, text in pending]),
                len(pending),
                self.dimensions,
            )
            self.dimensions = vectors.shape[1]
        current_keys = {(t, i) for t, i, _ in docs}
        with self.conn:
            if not compatible:
                self.conn.execute("DELETE FROM embeddings")
            for doc_type, doc_id in existing.keys() - current_keys:
                self.conn.execute(
                    "DELETE FROM embeddings WHERE doc_type = ? AND doc_id = ?",
                    (doc_type, doc_id),
                )
            for (doc_type, doc_id, text), vec in zip(pending, vectors):
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                    (doc_type, doc_id, text, json.dumps(vec.tolist())),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO embedding_metadata VALUES (1,?,?)",
                (self.index_key, self.dimensions or 0),
            )

    def search(self, query: str, doc_type: str | None = None,
               top_k: int = 5) -> list[tuple[str, float]]:
        """Search query within optional doc_type, returning at most top_k cosine scores."""
        try:
            return self._search(query, doc_type, top_k)
        except (EmbeddingError, ValueError, TypeError):
            logger.warning("向量查询失败，已回退关键词检索 / Vector lookup failed; using keywords")
            return []

    def _search(self, query: str, doc_type: str | None, top_k: int) -> list[tuple[str, float]]:
        """Validate cache identity, then rank query matches within doc_type up to top_k."""
        metadata = self.conn.execute(
            "SELECT index_key, dimensions FROM embedding_metadata WHERE id = 1",
        ).fetchone()
        if metadata is None or tuple(metadata) != (self.index_key, self.dimensions or 0):
            raise EmbeddingError("向量缓存身份不匹配")
        sql = "SELECT doc_id, vector FROM embeddings"
        params: tuple = ()
        if doc_type:
            sql += " WHERE doc_type = ?"
            params = (doc_type,)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return []
        query_vec = normalize_vectors(self.embedder.embed([query]), 1, self.dimensions)[0]
        matrix = normalize_vectors([json.loads(r[1]) for r in rows], len(rows), self.dimensions)
        scores = matrix @ query_vec  # 已归一化,点积即余弦
        ranked = sorted(zip((r[0] for r in rows), scores.tolist()),
                        key=lambda x: -x[1])
        return ranked[:top_k]


def rrf_fuse(keyword_ids: list[str], vector_ids: list[str],
             k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion 混合两路排序。"""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(keyword_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


def create_vector_store(
    conn: sqlite3.Connection,
    model_path: str,
    cloud: CloudEmbeddingSettings | None = None,
) -> VectorStore | None:
    """Build on conn using cloud when configured, otherwise optional local model_path."""
    if cloud is None and not Path(model_path).exists():
        return None
    embedder = None
    try:
        if cloud is not None:
            embedder = OpenAIEmbedder(cloud)
        return VectorStore(conn, model_path, embedder)
    except Exception:
        if embedder is not None:
            embedder.close()
        logger.warning("向量索引不可用，已回退关键词检索 / Vector index unavailable; using keywords")
        return None
