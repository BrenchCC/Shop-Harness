"""全局配置与 token 估算。"""

from __future__ import annotations

from pydantic import BaseModel


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
