"""trajectory 评测执行器:跑 scenarios.py 全部场景,输出通过/失败矩阵。

用法: .venv/bin/python eval/run_eval.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopharness.cli import build_harness  # noqa: E402
from shopharness.config import Settings  # noqa: E402
from shopharness.llm.mock_client import MockLLM  # noqa: E402

from scenarios import SCENARIOS, Scenario  # noqa: E402


def run_scenario(sc: Scenario, tmpdir: str) -> list[str]:
    """返回失败原因列表,空列表表示通过。"""
    overrides = {"db_path": f"{tmpdir}/{sc.name}.db", "trace_dir": f"{tmpdir}/traces",
                 "rag_enabled": False,  # 评测用 Mock,语义检索单测覆盖
                 **sc.settings_overrides}
    settings = Settings(**overrides)
    harness = build_harness(settings, MockLLM())
    for sql in sc.preset_sql:
        harness.conn.execute(sql)
    harness.conn.commit()

    tool_calls: list[str] = []
    event_types: list[str] = []
    last_reply = ""
    handed_off = False
    for turn in sc.turns:
        result = harness.handle(turn)
        last_reply = result.reply
        handed_off = handed_off or result.handed_off
        for event in result.events:
            event_types.append(event.type)
            if event.type == "tool_call":
                tool_calls.append(event.detail.split("(")[0])

    failures: list[str] = []

    # 工具调用顺序(子序列匹配)
    idx = 0
    for expected in sc.expect_tools:
        while idx < len(tool_calls) and tool_calls[idx] != expected:
            idx += 1
        if idx == len(tool_calls):
            failures.append(f"缺少工具调用 {expected}(实际:{tool_calls})")
        else:
            idx += 1
    for et in sc.expect_events:
        if et not in event_types:
            failures.append(f"缺少事件 {et}(实际:{sorted(set(event_types))})")
    for et in sc.forbid_events:
        if et in event_types:
            failures.append(f"出现禁止事件 {et}")
    for kw in sc.reply_contains:
        if kw not in last_reply:
            failures.append(f"回复缺少关键词「{kw}」(实际:{last_reply[:80]}…)")
    if handed_off != sc.expect_handoff:
        failures.append(f"转人工断言失败:期望 {sc.expect_handoff},实际 {handed_off}")
    for key in sc.expect_facts:
        if key not in harness.context.state.facts:
            failures.append(f"L2 事实缺少「{key}」(实际:{harness.context.state.facts})")
    for sql, params, expected in sc.db_checks:
        actual = harness.conn.execute(sql, params).fetchone()[0]
        if actual != expected:
            failures.append(f"DB 断言失败:{sql} → {actual},期望 {expected}")
    return failures


def main() -> int:
    gate_mode = "--gate" in sys.argv
    with tempfile.TemporaryDirectory() as tmpdir:
        passed = 0
        if not gate_mode:
            print(f"{'场景':<28} 结果")
            print("-" * 60)
        for sc in SCENARIOS:
            failures = run_scenario(sc, tmpdir)
            if failures:
                print(f"{sc.name:<28} ❌")
                for f in failures:
                    print(f"    - {f}")
            else:
                if not gate_mode:
                    print(f"{sc.name:<28} ✅")
                passed += 1
        print("-" * 60)
        print(f"通过 {passed}/{len(SCENARIOS)}")
        return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(main())
