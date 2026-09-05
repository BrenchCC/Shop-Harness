"""LoRA 前后对比评测:留出场景(不在训练集里),对比基座与微调后模型。

指标(trajectory 口径):
- 期望工具被调用(子序列匹配)
- 无 correction / tool_error / handoff 事件

用法:
    .venv/bin/python evolve/eval_lora.py --model Qwen/Qwen3-8B-FP8          # 基线
    .venv/bin/python evolve/eval_lora.py --model cs-lora --served-name cs-lora
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopharness.cli import build_harness  # noqa: E402
from shopharness.config import Settings  # noqa: E402
from shopharness.llm.openai_client import OpenAIClient  # noqa: E402

# 留出场景:与 collect_sft.py 训练脚本无重叠
HELD_OUT: list[tuple[str, list[str], list[str]]] = [
    ("送礼咖啡咨询", ["有没有适合送人的咖啡"], ["search_products"]),
    ("待付款订单查询", ["订单 20260701003 是什么状态"], ["get_order"]),
    ("精华到手价", ["YX-9002 到手价多少"], ["calc_discount"]),
    ("发货时效问答", ["你们发货要多久"], ["search_faq"]),
    ("不存在商品", ["投影仪有吗"], ["search_products"]),
    ("暖杯垫咨询", ["冬天办公室想喝热水,有什么神器"], ["search_products"]),
]

BAD_EVENTS = {"correction", "tool_error", "circuit_break", "handoff"}


def run(model: str, endpoint: str) -> tuple[int, list[str]]:
    llm = OpenAIClient(base_url=endpoint, model=model)
    passed = 0
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (name, turns, expect_tools) in enumerate(HELD_OUT):
            settings = Settings(db_path=f"{tmpdir}/h{idx}.db",
                                trace_dir=f"{tmpdir}/traces",
                                rag_enabled=True)
            harness = build_harness(settings, llm, session_id=f"eval-{idx}")
            ok = True
            called: list[str] = []
            for turn in turns:
                result = harness.handle(turn)
                event_types = {e.type for e in result.events}
                called += [e.detail.split("(")[0] for e in result.events
                           if e.type == "tool_call"]
                if event_types & BAD_EVENTS:
                    ok = False
            missing = [t for t in expect_tools if t not in called]
            if missing:
                ok = False
            passed += ok
            lines.append(f"  {'✅' if ok else '❌'} {name}"
                         f"(调用:{called},缺失:{missing or '无'})")
    return passed, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    args = parser.parse_args()
    passed, lines = run(args.model, args.endpoint)
    print(f"模型: {args.model}")
    for line in lines:
        print(line)
    print(f"通过 {passed}/{len(HELD_OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
