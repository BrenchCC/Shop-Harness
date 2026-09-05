"""SFT 轨迹采集:用真实 vLLM 模型跑脚本化买家会话,拒绝采样导出训练数据。

数据飞轮最后一步的前置:不是用 Mock 造假数据,而是让真实模型在 harness 里
跑真实工具调用,只保留成功回合(无 handoff/无护栏拒绝/无工具错误),
每个回合导出一条"完整上下文 → 模型响应"的 OpenAI messages 格式样本。

用法:
    .venv/bin/python evolve/collect_sft.py [--endpoint http://localhost:8000/v1]
输出: evolve/out/sft_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopharness.cli import build_harness  # noqa: E402
from shopharness.config import Settings  # noqa: E402
from shopharness.llm.base import Message  # noqa: E402
from shopharness.llm.openai_client import OpenAIClient  # noqa: E402

from evolve.export_sft import mask_pii  # noqa: E402

# 脚本化买家会话:覆盖各工具与技能,每段会话 1-3 轮
SCRIPTS: list[list[str]] = [
    ["有什么降噪耳机推荐吗"],
    ["我想买个机械键盘,有推荐吗"],
    ["办公室用的静音鼠标有吗"],
    ["最近睡觉脖子疼,有什么好物"],
    ["YX-1001 现在到手价多少"],
    ["冲锋衣有优惠吗"],
    ["帮我查一下订单 20260701001 的状态"],
    ["订单 20260701001 发的什么快递,到哪了"],
    ["我的订单 20260701002 物流到哪里了"],
    ["买错了能退吗"],
    ["你们可以开发票吗"],
    ["耳机保修政策是什么"],
    ["优惠券能叠加用吗"],
    ["你们的商品是正品吗"],
    ["YX-1001 和 YX-1002 对比哪个好"],
    ["帮我处理退货,订单 20260701002"],
    ["有订单还没付款,帮我催一下"],
    ["瑜伽垫和跳绳,新手买哪个"],
    ["儿童绘本适合 3 岁孩子吗"],
    ["洗面奶敏感肌能用吗"],
    # 两轮会话:考察上下文延续
    ["有降噪耳机推荐吗", "它有优惠吗"],
    ["帮我查一下订单 20260701001", "它什么时候能发货"],
    ["冲锋衣怎么样", "帮我算一下到手价"],
    ["有什么枕头推荐", "和四件套一起买有优惠吗"],
    ["气泡水多少钱", "买两箱有优惠吗"],
]


def to_openai(m: Message) -> dict:
    return OpenAIClient._to_openai(m)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=Settings().model)
    parser.add_argument("--out", default="evolve/out/sft_full.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llm = OpenAIClient(base_url=args.endpoint, model=args.model)

    n_samples = n_skipped = 0
    with tempfile.TemporaryDirectory() as tmpdir, \
            out_path.open("w", encoding="utf-8") as fout:
        for idx, script in enumerate(SCRIPTS):
            settings = Settings(db_path=f"{tmpdir}/s{idx}.db",
                                trace_dir=f"{tmpdir}/traces",
                                rag_enabled=True)
            harness = build_harness(settings, llm,
                                    session_id=f"sft-{idx:03d}",
                                    buyer_id=f"buyer-{idx}")
            for turn in script:
                before = len(harness.context.history)
                result = harness.handle(turn)
                bad_events = {"handoff", "guardrail_denied", "circuit_break",
                              "correction"}
                if any(e.type in bad_events for e in result.events):
                    n_skipped += 1
                    continue
                # 样本 = 当前 system + 本回合新增消息(用户输入 → 工具 → 回复)
                system_msg = harness.context.build(
                    harness.skills.route(turn))[0]
                delta = harness.context.history[before:]
                messages = [to_openai(system_msg)] + \
                    [to_openai(m) for m in delta]
                for m in messages:
                    if m.get("content"):
                        m["content"] = mask_pii(m["content"])
                fout.write(json.dumps({"messages": messages},
                                      ensure_ascii=False) + "\n")
                n_samples += 1
            print(f"[{idx + 1}/{len(SCRIPTS)}] {script[0][:20]}… "
                  f"累计样本 {n_samples}")
        print(f"\n完成:{n_samples} 条样本,跳过 {n_skipped} 条(含异常事件)")
        print(f"输出:{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
