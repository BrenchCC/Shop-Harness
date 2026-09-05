"""上下文工程:分层上下文(L0-L4)+ 三级 compaction。

分层(对应 DESIGN.md §3.2):
- L0 system:角色/红线/激活技能指令(每轮重组)
- L2 结构化会话状态:意图、已确认事实、待办(渲染进 system)
- L3 会话历史 + L4 工具结果:统一存放在 history,超预算逐级压缩

三级 compaction(由 Harness 触发,对模型透明):
1. 工具结果瘦身:旧的长工具结果替换为摘要 + artifact 引用
2. 滑窗 + 事实抽取:丢弃中间轮原文前,先抽取关键事实写入 L2
3. 全量摘要:整体压缩为摘要段,仅保留最近若干条原文
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings, estimate_tokens
from ..llm.base import LLMClient, Message
from .skills import Skill

BASE_PROMPT = """你是电商店铺的 AI 客服,正在服务买家咨询。

红线:
- 只使用工具返回的真实数据,禁止编造商品参数、库存、价格与物流信息
- 改价、退款等危险操作必须先获得买家明确确认
- 不知道的问题诚实说明,必要时建议转人工
风格:简洁、专业、友好,回复控制在 150 字以内。"""

EXTRACT_PROMPT = """请从以下对话片段中抽取关键事实,每行一条,格式「事实名: 值」。
只抽取:买家看中的商品 SKU、涉及的订单号、买家明确确认过的事项、价格承诺。
没有则输出「无」。"""

SUMMARY_PROMPT = """请将以下对话压缩为 100 字以内的摘要,保留:买家诉求、已确认事实、
已执行的操作、当前进展。直接输出摘要文本。"""


class SessionState(BaseModel):
    """L2:结构化会话状态,compaction 时事实只进不出。"""

    intent: str | None = None
    facts: dict[str, str] = Field(default_factory=dict)
    pending: list[str] = Field(default_factory=list)
    active_skills: list[str] = Field(default_factory=list)


class ArtifactStore:
    """被瘦身的工具结果原文存取(可替换为磁盘/对象存储)。"""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._counter = 0

    def put(self, content: str) -> str:
        self._counter += 1
        ref = f"artifact-{self._counter}"
        self._store[ref] = content
        return ref

    def get(self, ref: str) -> str | None:
        return self._store.get(ref)


class CompactionEvent(BaseModel):
    level: int
    detail: str
    freed_tokens: int = 0


class ContextManager:
    def __init__(self, settings: Settings, base_prompt: str = BASE_PROMPT):
        self.settings = settings
        self.base_prompt = base_prompt
        self.history: list[Message] = []
        self.state = SessionState()
        self.artifacts = ArtifactStore()
        self.events: list[CompactionEvent] = []
        self.l1_context: str = ""  # L1:记忆/业务知识注入(会话开始时写入)

    # ------------------------------------------------------------ 构建

    def build(self, active_skills: list[Skill]) -> list[Message]:
        """组装发给模型的消息:L0(+L2)system + L3/L4 history。"""
        parts = [self.base_prompt]
        for skill in active_skills:
            parts.append(f"\n[当前技能: {skill.name}]\n{skill.instructions}")
        if self.l1_context:
            parts.append(f"\n[买家记忆]\n{self.l1_context}")
        state_json = json.dumps(
            self.state.model_dump(exclude={"active_skills"}),
            ensure_ascii=False)
        parts.append(f"\n[会话状态] {state_json}")
        return [Message.system("".join(parts))] + list(self.history)

    def history_tokens(self) -> int:
        return sum(estimate_tokens(m.content or "") +
                   (20 if m.tool_calls else 0) for m in self.history)

    # ------------------------------------------------------------ compaction

    def compact(self, llm: LLMClient) -> list[CompactionEvent]:
        new_events: list[CompactionEvent] = []
        new_events += self._slim_tool_results()
        if self.history_tokens() > self.settings.context_budget:
            event = self._window_with_fact_extract(llm)
            if event:
                new_events.append(event)
        if self.history_tokens() > self.settings.context_budget:
            event = self._full_summary(llm)
            if event:
                new_events.append(event)
        self.events.extend(new_events)
        return new_events

    def _slim_tool_results(self) -> list[CompactionEvent]:
        """一级:工具结果瘦身(保留最近 N 条原文)。"""
        tool_idx = [i for i, m in enumerate(self.history) if m.role == "tool"]
        protected = set(tool_idx[-self.settings.keep_recent_tool_results:])
        freed = 0
        count = 0
        for i, m in enumerate(self.history):
            if m.role == "tool" and i not in protected and m.content and \
                    len(m.content) > self.settings.tool_result_slim_chars:
                before = estimate_tokens(m.content)
                ref = self.artifacts.put(m.content)
                m.content = (f"[已归档] 工具 {m.name} 的完整结果(ref={ref}),"
                             f"摘要:{m.content[:80]}…")
                freed += before - estimate_tokens(m.content)
                count += 1
        if count:
            return [CompactionEvent(
                level=1, detail=f"瘦身 {count} 条历史工具结果",
                freed_tokens=freed)]
        return []

    def _window_with_fact_extract(self, llm: LLMClient) -> CompactionEvent | None:
        """二级:滑窗,丢弃中间轮前抽取关键事实写入 L2。"""
        n_head, n_tail = 2, self.settings.keep_recent_turns
        if len(self.history) <= n_head + n_tail:
            return None
        head = self.history[:n_head]
        tail = self.history[-n_tail:]
        middle = self.history[n_head:-n_tail]
        facts = self._extract_facts(llm, middle)
        self.state.facts.update(facts)
        freed = sum(estimate_tokens(m.content or "") for m in middle)
        self.history = head + tail
        return CompactionEvent(
            level=2,
            detail=f"滑窗丢弃 {len(middle)} 条中间消息,抽取事实 {len(facts)} 条",
            freed_tokens=freed)

    def _full_summary(self, llm: LLMClient) -> CompactionEvent | None:
        """三级:全量摘要,仅保留最近几条原文。"""
        n_tail = self.settings.keep_tail_after_summary
        if len(self.history) <= n_tail + 1:
            return None
        old = self.history[:-n_tail]
        tail = self.history[-n_tail:]
        summary = self._summarize(llm, old)
        freed = sum(estimate_tokens(m.content or "") for m in old)
        self.history = [Message.system(f"[早前会话摘要] {summary}")] + tail
        return CompactionEvent(
            level=3, detail=f"全量摘要压缩 {len(old)} 条消息",
            freed_tokens=freed)

    # ------------------------------------------------------------ LLM 内部调用

    def _extract_facts(self, llm: LLMClient,
                       messages: list[Message]) -> dict[str, str]:
        resp = llm.chat([Message.system(EXTRACT_PROMPT)] + messages)
        return _parse_fact_lines(resp.content or "")

    def _summarize(self, llm: LLMClient, messages: list[Message]) -> str:
        resp = llm.chat([Message.system(SUMMARY_PROMPT)] + messages)
        return (resp.content or "").strip() or "(摘要生成失败)"


def _parse_fact_lines(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-·* ")
        if not line or line == "无" or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key and value:
            facts[key] = value
    return facts
