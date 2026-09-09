"""全局配置与 token 估算。"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, SecretStr


class CloudLLMSettings(BaseModel):
    """Cloud credentials, independent of local vLLM settings."""

    base_url: str
    api_key: SecretStr
    model: str


class CloudEmbeddingSettings(BaseModel):
    """Embedding credentials, independent of conversation model settings."""

    base_url: str
    api_key: SecretStr
    model: str


def load_embedding_settings(env_path: str | Path = ".env") -> CloudEmbeddingSettings | None:
    """Load independent embedding credentials from env_path and process variables."""
    file_values = dotenv_values(env_path, interpolate = False)
    names = ("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL")
    alias = "EMBEDDDING_MODEL"
    # 兼容已有拼写，进程配置优先 / Accept the existing alias, process values first.
    values = {}
    for source in (file_values, os.environ):
        values.update({name: source[name] for name in names if name in source})
        if "EMBEDDING_MODEL" not in source and alias in source:
            values["EMBEDDING_MODEL"] = source[alias]
    if not values:
        return None
    config = {name: (values.get(name) or "").strip() for name in names}
    missing = [name for name in names if not config[name]]
    if missing:
        raise ValueError("云端向量配置缺失: " + ", ".join(missing))
    _validate_base_url(config["EMBEDDING_BASE_URL"], "EMBEDDING_BASE_URL")
    return CloudEmbeddingSettings(
        base_url = config["EMBEDDING_BASE_URL"],
        api_key = SecretStr(config["EMBEDDING_API_KEY"]),
        model = config["EMBEDDING_MODEL"],
    )


def _validate_base_url(value: str, name: str) -> None:
    """Validate URL value and identify invalid configuration by name, without echoing it."""
    try:
        url = urlsplit(value)
        valid = (url.scheme in ("http", "https") and url.hostname
                 and not url.username and not url.password
                 and not url.query and not url.fragment)
        url.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} 必须为不含凭据、查询参数或片段的 HTTP(S) 地址")


def load_cloud_settings(
    env_path: str | Path = ".env", model: str | None = None
) -> CloudLLMSettings | None:
    """Load cloud configuration without modifying the process environment.

    Args:
        env_path: Dotenv file in the working directory by default.
        model: Optional CLI model override.
    """
    # 显式读取且不污染本地模式 / Read explicitly without affecting local mode.
    values = {**dotenv_values(env_path, interpolate = False), **os.environ}
    names = ("LLM_BASE_URL", "LLM_API_KEY", "MODEL")
    if not any(name in values for name in names):
        return None
    config = {name: (values.get(name) or "").strip() for name in names}
    if model is not None:
        config["MODEL"] = model.strip()
    missing = [name for name in names if not config[name]]
    if missing:
        raise ValueError("云端配置缺失: " + ", ".join(missing))
    _validate_base_url(config["LLM_BASE_URL"], "LLM_BASE_URL")
    return CloudLLMSettings(
        base_url = config["LLM_BASE_URL"],
        api_key = SecretStr(config["LLM_API_KEY"]),
        model = config["MODEL"],
    )


class Settings(BaseModel):
    """Harness 可调参数;测试里用小预算实例化以触发 compaction。"""

    # 上下文(L3+L4 历史)估算 token 预算,超出即逐级触发 compaction
    context_budget: int = 4096
    max_tool_steps: int = 8                # 单轮用户消息内的最大工具步数
    keep_recent_tool_results: int = 4      # 一级瘦身保护的最近工具结果条数
    tool_result_slim_chars: int = 400      # 超过该长度的历史工具结果才会被瘦身
    keep_recent_turns: int = 6             # 二级滑窗保护的最近消息条数
    keep_tail_after_summary: int = 4       # 三级全量摘要后保留的原文条数
    circuit_breaker_failures: int = 2      # 同一工具连续失败熔断阈值
    max_corrections: int = 1               # 非法工具调用的自我纠正机会次数

    db_path: str = "shopharness/data/shop.db"
    skills_dir: str = "skills"
    trace_dir: str = "traces"

    # RAG:向量检索(bge-small-zh);模型缺失时自动降级为纯关键词检索
    rag_enabled: bool = True
    embedding_model: str = "models/bge-small-zh-v1.5"
    cloud_embedding: CloudEmbeddingSettings | None = None

    # vLLM(OpenAI-compatible)接入参数
    model: str = "Qwen/Qwen3-8B-FP8"
    base_url: str = "http://localhost:8000/v1"
    enable_thinking: bool = False          # /no_think:客服场景优先低延迟


def estimate_tokens(text: str) -> int:
    """启发式 token 估算:CJK 字符约 1 token,其余约 0.34 token。

    与真实 tokenizer 有偏差,但对预算触发判断足够稳定;
    生产环境可替换为 transformers AutoTokenizer。
    """
    total = 0.0
    for ch in text:
        total += 1.0 if ord(ch) > 0x2E7F else 0.34
    return max(1, int(total))
