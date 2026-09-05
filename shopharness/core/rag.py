"""RAG 检索增强:bge-small-zh 向量语义检索 + 关键词检索,RRF 混合排序。

设计要点:
- 只用 transformers 直读 bge 模型(mean pooling + L2 归一化),不引 sentence-transformers
- 向量存 SQLite embeddings 表,启动时增量构建;模型缺失时优雅降级为纯关键词检索
- CPU 推理(bge-small 单次毫秒级),不占用 vLLM 显存
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    doc_type TEXT NOT NULL,      -- product / faq
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    vector TEXT NOT NULL,        -- JSON 数组
    PRIMARY KEY (doc_type, doc_id)
);
"""
# FAQ 表结构与种子数据见 data/seed.py(业务数据统一入口)


class Embedder:
    """bge-small-zh 文本向量(mean pooling + 归一化),CPU 推理。"""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
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

    def __init__(self, conn: sqlite3.Connection, model_path: str):
        self.conn = conn
        self.conn.executescript(SCHEMA)
        self.embedder = Embedder(model_path)
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
        existing = {r[0:2] for r in self.conn.execute(
            "SELECT doc_type, doc_id FROM embeddings").fetchall()}
        pending = [(t, i, x) for t, i, x in self._doc_texts()
                   if (t, i) not in existing]
        if not pending:
            return
        vectors = self.embedder.embed([x for _, _, x in pending])
        for (doc_type, doc_id, text), vec in zip(pending, vectors):
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                (doc_type, doc_id, text, json.dumps(vec.tolist())))
        self.conn.commit()

    def search(self, query: str, doc_type: str | None = None,
               top_k: int = 5) -> list[tuple[str, float]]:
        """向量检索,返回 (doc_id, cosine 相似度) 降序。"""
        sql = "SELECT doc_id, vector FROM embeddings"
        params: tuple = ()
        if doc_type:
            sql += " WHERE doc_type = ?"
            params = (doc_type,)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return []
        query_vec = self.embedder.embed([query])[0]
        matrix = np.array([json.loads(r[1]) for r in rows])
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


def create_vector_store(conn: sqlite3.Connection,
                        model_path: str) -> VectorStore | None:
    """模型缺失/加载失败时优雅降级(返回 None,退回纯关键词检索)。"""
    if not Path(model_path).exists():
        return None
    try:
        return VectorStore(conn, model_path)
    except Exception:
        return None
