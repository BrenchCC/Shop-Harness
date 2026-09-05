"""子代理:上下文隔离的委托执行(对应 DESIGN.md §3.5)。

- 子代理复用 Harness 执行引擎,但持独立 ContextManager:
  中间检索噪音(几十个候选商品、多跳工具结果)不进入主对话上下文
- 只把最终结论摘要回传主代理,这是小上下文(8B/32K)模型做多跳检索的关键
- 子代理运行过程写入同一 trace(agent 字段区分),事件以 subagent_* 进入主流
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..config import Settings
from ..llm.base import LLMClient
from ..tools.registry import Tool, ToolRegistry
from .context import ContextManager
from .harness import Harness
from .hooks import HookBus
from .permissions import Level, PermissionManager
from .skills import SkillManager
from .trace import Tracer

RESEARCH_TOOLS = {"search_products", "search_faq", "get_product_detail",
                  "calc_discount"}
RESEARCH_PROMPT = """你是检索子代理,为主客服代理完成商品检索/比价任务。
用工具获取真实数据后,输出 150 字以内的结论摘要:候选商品、关键参数、
价格/到手价、你的推荐及理由。不要寒暄,直接给结论。"""

AFTERSALE_TOOLS = {"get_order", "get_logistics", "search_faq", "create_ticket"}
AFTERSALE_PROMPT = """你是售后工单子代理,为主客服代理处理退换货诉求。
按 SOP 执行:核实订单 → 查看物流 → 判断售后期限(签收 7 天内)→ 需要时建工单。
输出结论摘要:订单状态、判断依据、已执行操作、给买家的建议方案。"""


class SubagentRunner:
    def __init__(self, llm: LLMClient, registry: ToolRegistry,
                 settings: Settings, tracer: Tracer, conn: sqlite3.Connection):
        self.llm = llm
        self.registry = registry
        self.settings = settings
        self.tracer = tracer
        self.conn = conn

    def run(self, agent_name: str, task: str, tool_names: set[str],
            system_prompt: str) -> dict[str, Any]:
        """在隔离上下文中跑一个子代理,返回结论摘要。"""
        self.tracer.span("subagent_call", agent=agent_name, task=task)
        sub_settings = self.settings.model_copy(update={"max_tool_steps": 4})
        sub_harness = Harness(
            llm=self.llm,
            registry=self.registry,
            hooks=HookBus(),
            permissions=PermissionManager(),
            skills=SkillManager(skills_dir="__none__"),  # 子代理不加载技能
            context=ContextManager(sub_settings, base_prompt=system_prompt),
            tracer=self.tracer,
            settings=sub_settings,
            conn=self.conn,
            tool_whitelist=tool_names,
        )
        result = sub_harness.handle(task)
        steps = sum(1 for e in result.events if e.type == "tool_call")
        self.tracer.span("subagent_result", agent=agent_name, steps=steps)
        return {"agent": agent_name, "summary": result.reply,
                "tool_steps": steps}


def register_subagent_tools(registry: ToolRegistry,
                            runner: SubagentRunner) -> None:
    """把子代理注册为主代理的两个委托工具。"""
    obj = {"type": "object"}
    registry.register(Tool(
        "delegate_research",
        "将复杂的商品检索/比价/优惠计算任务委托给检索子代理,返回结论摘要。"
        "适合需要多次检索对比的场景",
        {**obj, "properties": {
            "query": {"type": "string", "description": "检索任务描述"}},
         "required": ["query"]},
        Level.READ,
        lambda query: runner.run("research", query, RESEARCH_TOOLS,
                                 RESEARCH_PROMPT)))
    registry.register(Tool(
        "delegate_aftersale",
        "将退换货诉求委托给售后工单子代理按 SOP 处理,返回结论摘要",
        {**obj, "properties": {
            "issue": {"type": "string", "description": "买家诉求描述"}},
         "required": ["issue"]},
        Level.READ,
        lambda issue: runner.run("aftersale", issue, AFTERSALE_TOOLS,
                                 AFTERSALE_PROMPT)))
