"""OpenAI 兼容云端向量客户端 / No local inference packages required."""

from __future__ import annotations

import os
import sys
import json
import hashlib

import numpy as np
from openai import OpenAI

sys.path.append(os.getcwd())

from shopharness.config import CloudEmbeddingSettings


class EmbeddingError(RuntimeError):
    """Sanitized embedding service or vector validation error."""


def normalize_vectors(
    vectors: list | np.ndarray,
    count: int,
    dimensions: int | None = None,
) -> np.ndarray:
    """Validate vectors against count and optional dimensions, then L2-normalize."""
    try:
        matrix = np.asarray(vectors, dtype = np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] == 0:
            raise ValueError
        if dimensions is not None and matrix.shape[1] != dimensions:
            raise ValueError
        norms = np.linalg.norm(matrix, axis = 1, keepdims = True)
        if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError
        return (matrix / norms).astype(np.float32)
    except (ValueError, TypeError, OverflowError):
        raise EmbeddingError("向量数量、维度或数值无效") from None


class OpenAIEmbedder:
    def __init__(self, settings: CloudEmbeddingSettings):
        """Initialize the independent endpoint, model and credentials from settings."""
        self.model = settings.model
        self.dimensions: int | None = None
        identity = ["openai-float-l2-v1", settings.base_url.rstrip("/"), self.model]
        self.index_key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.client = OpenAI(
            base_url = settings.base_url,
            api_key = settings.api_key.get_secret_value(),
            timeout = 30.0,
            max_retries = 0,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches of at most ten, preserving input order."""
        if not texts:
            return np.empty((0, self.dimensions or 0), dtype = np.float32)
        batches = []
        # 百炼 v3/v4 每批最多 10 条 / Conservative batch size for Bailian models.
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            try:
                result = self.client.embeddings.create(
                    model = self.model,
                    input = batch,
                    encoding_format = "float",
                )
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                detail = f"HTTP {status}" if isinstance(status, int) else type(exc).__name__
                raise EmbeddingError(f"向量 API 请求失败 ({detail})") from None
            try:
                rows = sorted(result.data, key = lambda row: row.index)
                if [row.index for row in rows] != list(range(len(batch))):
                    raise ValueError
                vectors = [row.embedding for row in rows]
            except (ValueError, TypeError, AttributeError):
                raise EmbeddingError("向量 API 返回的输入索引无效") from None
            matrix = normalize_vectors(vectors, len(batch), self.dimensions)
            self.dimensions = matrix.shape[1]
            batches.append(matrix)
        return np.concatenate(batches)

    def close(self) -> None:
        """Release the HTTP connection pool."""
        self.client.close()
