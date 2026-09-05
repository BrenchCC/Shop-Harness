"""演示对话 REPL。

用法:
    python -m shopharness.cli --mock                      # Mock 模式(默认,零依赖)
    python -m shopharness.cli --endpoint http://localhost:8000/v1  # 真实 vLLM
    printf '有耳机推荐吗\\n退出\\n' | python -m shopharness.cli --mock
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .core.context import ContextManager
from .core.harness import Harness, TurnResult
from .core.hooks import HookBus
from .core.permissions import PermissionManager
from .core.skills import SkillManager
from .core.trace import Tracer
from .data.seed import ensure_db
from .llm.base import LLMClient
from .tools.servers import (build_registry, make_audit_hook,
                             make_price_guardrail)


def build_harness(settings: Settings, llm: LLMClient,
                  session_id: str | None = None,
                  buyer_id: str = "anonymous") -> Harness:
    """组装完整 Harness(依赖注入入口,测试与 eval 也走这里)。"""
    from .core.memory import MemoryStore
    from .core.rag import create_vector_store
    from .core.subagent import SubagentRunner, register_subagent_tools

    conn = ensure_db(settings.db_path)
    vector_store = None
    if settings.rag_enabled:
        vector_store = create_vector_store(conn, settings.embedding_model)
    registry = build_registry(conn, vector_store)
    hooks = HookBus(pre_hooks=[make_price_guardrail(conn)],
                    post_hooks=[make_audit_hook(conn)])
    skills = SkillManager(skills_dir=settings.skills_dir)
    skills.load()
    tracer = Tracer(settings.trace_dir, session_id=session_id)
    context = ContextManager(settings)
    # M3a:子代理工具注册进主工具表(上下文隔离的委托执行)
    runner = SubagentRunner(llm, registry, settings, tracer, conn)
    register_subagent_tools(registry, runner)
    return Harness(llm=llm, registry=registry, hooks=hooks,
                   permissions=PermissionManager(), skills=skills,
                   context=context, tracer=tracer, settings=settings, conn=conn,
                   memory=MemoryStore(conn), buyer_id=buyer_id)


def make_llm(args: argparse.Namespace, settings: Settings) -> LLMClient:
    if args.endpoint:
        from .llm.openai_client import OpenAIClient
        return OpenAIClient(base_url=args.endpoint, model=args.model,
                            enable_thinking=args.thinking)
    from .llm.mock_client import MockLLM
    return MockLLM()


def print_result(result: TurnResult, verbose: bool) -> None:
    for event in result.events:
        icons = {"tool_call": "🔧", "tool_result": "✅", "tool_error": "⚠️",
                 "dangerous_intercepted": "🛑", "guardrail_denied": "🛡️",
                 "correction": "♻️", "circuit_break": "🔥",
                 "compaction": "🗜️", "handoff": "👤",
                 "skill_activated": "🎯", "confirmed": "👍",
                 "memory_injected": "🧠"}
        print(f"  {icons.get(event.type, '•')} [{event.type}] {event.detail}")
    print(f"客服: {result.reply}\n")


def run_flow(settings: Settings) -> None:
    """售后长流程演示:LangGraph + checkpoint,支持中断后恢复。

    用法:先输入订单号发起流程;流程在「方案确认」处挂起,可直接回复,
    也可以 Ctrl-D 退出后再次运行本命令 —— 会从 checkpoint 恢复继续。
    """
    from .flows.aftersale import AftersaleFlow
    ensure_db(settings.db_path)
    flow = AftersaleFlow(settings.db_path)
    print("售后流程 demo(LangGraph,支持跨进程恢复)")
    thread_id = input("会话 ID(回车默认 cli): ").strip() or "cli"

    pending = flow.pending_question(thread_id)
    if pending:
        print(f"[已从 checkpoint 恢复] 待确认方案:\n{pending}")
    else:
        order_id = input("订单号: ").strip()
        view = flow.start(thread_id, order_id)
        if view["status"] == "done":
            print(view["result"])
            return
        print(f"方案:\n{view['question']}")

    while True:
        try:
            answer = input("买家答复: ").strip()
        except EOFError:
            print("\n流程已挂起,再次运行本命令可从断点恢复。")
            return
        if not answer:
            continue
        view = flow.resume(thread_id, answer)
        print(view["result"])
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="ShopHarness 客服演示")
    parser.add_argument("--mock", action="store_true", help="Mock 模式(默认)")
    parser.add_argument("--endpoint", help="vLLM OpenAI endpoint")
    parser.add_argument("--model", default=Settings().model)
    parser.add_argument("--thinking", action="store_true",
                        help="开启 Qwen3 思考模式")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--session", help="会话 ID(用于 trace 文件命名)")
    parser.add_argument("--buyer", default="buyer-demo",
                        help="买家 ID(记忆按此维度沉淀)")
    parser.add_argument("--flow", choices=["aftersale"],
                        help="进入长流程演示(售后工单)")
    args = parser.parse_args()

    settings = Settings()
    if args.flow:
        run_flow(settings)
        return

    harness = build_harness(settings, make_llm(args, settings), args.session,
                            buyer_id=args.buyer)
    mode = f"vLLM({args.endpoint})" if args.endpoint else "Mock"
    print(f"ShopHarness 客服 demo [{mode} 模式] — 输入买家消息,Ctrl-D 退出")
    print(f"trace: {harness.tracer.path}\n")

    while True:
        try:
            user = input("买家: ").strip()
        except EOFError:
            print("\n会话结束。")
            break
        if not user:
            continue
        if user in ("退出", "exit", "quit"):
            print("会话结束。")
            break
        result = harness.handle(user)
        print_result(result, args.verbose)
        if args.verbose:
            print(f"  [tokens] 历史约 {harness.context.history_tokens()} / "
                  f"预算 {settings.context_budget}")
            print()

    harness.end_session()  # M4a:沉淀情景/语义记忆
    print(f"(会话摘要与买家画像已写入记忆库:{settings.db_path})")


if __name__ == "__main__":
    sys.exit(main())
