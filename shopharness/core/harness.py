"""Harness 主 loop:turn 管理、权限门、错误自愈、熔断、转人工。

一次用户消息 → 循环「模型决策 ↔ 工具执行」,直到模型给出最终回复。
所有事件进入 events(供 CLI/eval 断言)与 trace(供可观测性)。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..llm.base import LLMClient, LLMError, Message
from ..tools.registry import ToolError, ToolRegistry
from .context import ContextManager
from .handoff import build_summary, detect_handoff, handoff_reply
from .hooks import HookBus
from .memory import MemoryStore
from .permissions import PermissionManager
from .skills import SkillManager
from .trace import Tracer

CONFIRM_KEYWORDS = ["确认", "好的", "可以", "行", "同意", "对"]


@dataclass
class TurnEvent:
    type: str            # skill_activated / tool_call / tool_result / tool_error /
                         # dangerous_intercepted / guardrail_denied / correction /
                         # circuit_break / compaction / handoff / confirmed
    detail: str


@dataclass
class TurnResult:
    reply: str
    events: list[TurnEvent] = field(default_factory=list)
    handed_off: bool = False
    handoff_summary: str | None = None


class Harness:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, hooks: HookBus,
                 permissions: PermissionManager, skills: SkillManager,
                 context: ContextManager, tracer: Tracer, settings: Settings,
                 conn: sqlite3.Connection,
                 tool_whitelist: set[str] | None = None,
                 memory: MemoryStore | None = None,
                 buyer_id: str = "anonymous"):
        self.llm = llm
        self.registry = registry
        self.hooks = hooks
        self.permissions = permissions
        self.skills = skills
        self.context = context
        self.tracer = tracer
        self.settings = settings
        self.conn = conn
        self.forced_whitelist = tool_whitelist  # 子代理场景:强制工具集
        self.memory = memory
        self.buyer_id = buyer_id
        self._memory_loaded = False
        self._handed_off = False
        self.action_log: list[str] = []
        self._failures: dict[str, int] = {}
        self._corrections = 0
        self._pending_dangerous: tuple[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------ 入口

    def handle(self, user_text: str) -> TurnResult:
        events: list[TurnEvent] = []
        self.tracer.span("user_message", content=user_text)

        # M4a:首轮注入买家记忆到 L1
        if self.memory and not self._memory_loaded:
            self.context.l1_context = self.memory.build_l1_context(self.buyer_id)
            if self.context.l1_context:
                events.append(TurnEvent("memory_injected", self.buyer_id))
                self.tracer.span("memory_injected", buyer_id=self.buyer_id)
            self._memory_loaded = True

        # 0. 买家主动要求人工 → 直接转
        reason = detect_handoff(user_text)
        if reason:
            self.context.history.append(Message.user(user_text))
            return self._handoff(reason, events)

        # 1. 危险操作确认:有待确认项 + 买家回复确认词 → 放行该工具一次
        if self._pending_dangerous and any(
                kw in user_text for kw in CONFIRM_KEYWORDS):
            tool_name, _ = self._pending_dangerous
            self.permissions.confirm(tool_name)
            events.append(TurnEvent("confirmed", f"买家确认执行 {tool_name}"))
            self.context.history.append(Message.user(user_text))
            self.context.history.append(Message.system(
                f"[系统] 买家已明确确认执行 {tool_name},请继续完成该操作。"))
            self._pending_dangerous = None
        else:
            self.context.history.append(Message.user(user_text))

        # 2. 技能路由 → 工具集裁剪
        active = self.skills.route(user_text)
        self.context.state.active_skills = [s.name for s in active]
        if active:
            events.append(TurnEvent(
                "skill_activated", ",".join(s.name for s in active)))
        whitelist = self.forced_whitelist or self.skills.tool_whitelist(active)
        tool_schemas = self.registry.schemas(whitelist)

        # 3. 主循环
        for _step in range(self.settings.max_tool_steps):
            messages = self.context.build(active)
            try:
                resp = self.llm.chat(messages, tool_schemas)
            except LLMError as exc:
                self.tracer.span("llm_error", error=str(exc))
                return self._handoff(f"模型服务异常({exc})", events)

            self.context.history.append(resp)
            if not resp.tool_calls:
                events += self._compact()
                reply = resp.content or "(模型无回复)"
                self.tracer.span("assistant_reply", content=reply)
                return TurnResult(reply=reply, events=events)

            for call in resp.tool_calls:
                result = self._execute_tool(call.id, call.name,
                                            call.arguments, events)
                if result is not None:
                    return result  # 熔断/纠正超限 → 已转人工
            events += self._compact()

        return self._handoff("单轮处理步数超限", events)

    # ------------------------------------------------------------ 工具执行

    def _execute_tool(self, call_id: str, name: str, args: dict[str, Any],
                      events: list[TurnEvent]) -> TurnResult | None:
        """返回 None 表示继续循环;返回 TurnResult 表示已转人工终止。"""
        events.append(TurnEvent(
            "tool_call", f"{name}({json.dumps(args, ensure_ascii=False)})"))
        tool = self.registry.get(name)
        if tool is None:
            self._corrections += 1
            available = ",".join(self.registry.names())
            self.context.history.append(Message.tool(
                call_id, name,
                f"[系统纠正] 工具 {name} 不存在,可用工具:{available}。"
                "请使用可用工具或直接文字回复。"))
            events.append(TurnEvent("correction", f"非法工具名 {name}"))
            if self._corrections > self.settings.max_corrections:
                return self._handoff("模型多次调用非法工具", events)
            return None

        # 权限门(危险操作)
        decision = self.permissions.check(tool.level, name)
        if decision.need_confirm:
            self._pending_dangerous = (name, args)
            self._audit(name, args, {"intercepted": True})
            self.context.history.append(
                Message.tool(call_id, name, decision.reason or ""))
            events.append(TurnEvent("dangerous_intercepted", name))
            self.tracer.span("dangerous_intercepted", tool=name, args=args)
            return None

        # 护栏 pre hook
        hook_result = self.hooks.run_pre(name, args)
        if not hook_result.allow:
            self._audit(name, args, {"denied": hook_result.reason})
            self.context.history.append(Message.tool(
                call_id, name,
                f"[护栏拒绝] {hook_result.reason}。请向买家解释无法执行,给出替代方案。"))
            events.append(TurnEvent("guardrail_denied",
                                    hook_result.reason or name))
            self.tracer.span("guardrail_denied", tool=name,
                             reason=hook_result.reason)
            return None

        # 执行(硬失败/软失败都计入熔断)
        try:
            result = tool.execute(args)
        except Exception as exc:  # noqa: BLE001 — 工具异常统一熔断处理
            self._failures[name] = self._failures.get(name, 0) + 1
            self.context.history.append(Message.tool(
                call_id, name, f"[工具错误] {type(exc).__name__}: {exc}"))
            events.append(TurnEvent("tool_error", f"{name}: {exc}"))
            self.tracer.span("tool_error", tool=name, error=str(exc))
            if self._failures[name] >= self.settings.circuit_breaker_failures:
                events.append(TurnEvent("circuit_break", name))
                return self._handoff(f"工具 {name} 连续失败,已熔断", events)
            return None

        if "error" in result:
            self._failures[name] = self._failures.get(name, 0) + 1
            self.context.history.append(Message.tool(
                call_id, name, json.dumps(result, ensure_ascii=False)))
            events.append(TurnEvent("tool_error",
                                    f"{name}: {result['error']}"))
            if self._failures[name] >= self.settings.circuit_breaker_failures:
                events.append(TurnEvent("circuit_break", name))
                return self._handoff(f"工具 {name} 连续失败,已熔断", events)
            return None

        # 成功
        self._failures.pop(name, None)
        self.hooks.run_post(name, args, result)
        if tool.level.value in ("write", "dangerous"):
            self._audit(name, args, result)
        self.action_log.append(
            f"{name}({json.dumps(args, ensure_ascii=False)})")
        self.context.history.append(
            Message.tool(call_id, name, json.dumps(result, ensure_ascii=False)))
        events.append(TurnEvent("tool_result", f"{name} 成功"))
        self.tracer.span("tool_result", tool=name, args=args, ok=True)
        return None

    # ------------------------------------------------------------ 内部

    def _compact(self) -> list[TurnEvent]:
        events = []
        for ce in self.context.compact(self.llm):
            events.append(TurnEvent(
                "compaction", f"L{ce.level} {ce.detail} (释放约 {ce.freed_tokens} tokens)"))
            self.tracer.span("compaction", level=ce.level, detail=ce.detail)
        return events

    def _handoff(self, reason: str, events: list[TurnEvent]) -> TurnResult:
        self._handed_off = True
        summary = build_summary(self.context.state, self.action_log, reason)
        cur = self.conn.execute(
            "INSERT INTO tickets(issue_type, summary) VALUES (?,?)",
            ("转人工", summary))
        self.conn.commit()
        reply = handoff_reply(summary, cur.lastrowid)
        self.context.history.append(Message.assistant(reply))
        events.append(TurnEvent("handoff", reason))
        self.tracer.span("handoff", reason=reason, ticket_id=cur.lastrowid)
        return TurnResult(reply=reply, events=events, handed_off=True,
                          handoff_summary=summary)

    def _audit(self, tool: str, args: dict[str, Any],
               result: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO audit(tool, args, result) VALUES (?,?,?)",
            (tool, json.dumps(args, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False)))
        self.conn.commit()

    # ------------------------------------------------------------ M4a 记忆

    def end_session(self) -> None:
        """会话结束:LLM 摘要落情景记忆,规则蒸馏偏好落语义记忆。"""
        if not self.memory or not self.context.history:
            return
        summary = self.context._summarize(self.llm, self.context.history)
        self.memory.add_session_summary(self.buyer_id, summary)
        prefs = self.memory.distill_preferences(
            self.context.state.facts, self.action_log, self._handed_off)
        if prefs:
            self.memory.update_profile(self.buyer_id, prefs)
        self.tracer.span("session_end", buyer_id=self.buyer_id,
                         summary=summary, prefs=prefs)
