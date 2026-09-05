"""自进化闭环主流程:挖掘 → 提案 → (可选应用) → 门禁 → 灰度/回滚。

默认 dry-run:只产出 bad case 报告与提案,不改任何技能文件。
--apply:把提案应用到技能文件(实验性),门禁不过则自动回滚。

用法:
    python -m evolve.run_cycle                 # dry-run
    python -m evolve.run_cycle --apply         # 完整闭环(带自动回滚)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shopharness.llm.mock_client import MockLLM  # noqa: E402

from evolve import gate, mine_bad_cases, propose  # noqa: E402
from evolve.release import SkillRelease  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="自进化闭环")
    parser.add_argument("--apply", action="store_true",
                        help="实际应用提案(默认 dry-run)")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--db", default="shopharness/data/shop.db")
    parser.add_argument("--endpoint", help="用真实 vLLM 生成提案(默认 Mock)")
    args = parser.parse_args()

    # 1. 挖掘
    cases = mine_bad_cases.mine(args.trace_dir)
    if not cases:
        print("没有挖到 bad case,闭环结束。")
        return 0
    by_type: dict[str, int] = {}
    for c in cases:
        by_type[c.type] = by_type.get(c.type, 0) + 1
    print(f"[1/4] 挖出 {len(cases)} 条 bad case:{by_type}")

    # 2. 提案
    llm = MockLLM()
    if args.endpoint:
        from shopharness.llm.openai_client import OpenAIClient
        llm = OpenAIClient(base_url=args.endpoint)
    import json
    from dataclasses import asdict
    out = Path("evolve/out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bad_cases.json").write_text(
        json.dumps([asdict(c) for c in cases], ensure_ascii=False, indent=2),
        encoding="utf-8")
    proposals = propose.propose_all(str(out / "bad_cases.json"), "skills", llm)
    print(f"[2/4] 生成 {len(proposals)} 份提案(位于 evolve/out/proposals/)")

    if not args.apply:
        print("[dry-run] 不应用提案。审查提案后加 --apply 走完整闭环。")
        return 0

    # 3-4. 应用 → 门禁 → 灰度/回滚
    release = SkillRelease(args.db)
    for path in proposals:
        # 提案文件格式: 标题 + 正文;应用策略为把正文追加为技能补充指令
        skill_name = path.stem.split("-")[-1]
        # 从 TYPE_TO_SKILL 反查完整技能名
        for full_name in propose.TYPE_TO_SKILL.values():
            if path.stem.endswith(full_name):
                skill_name = full_name
                break
        skill_file = Path("skills") / skill_name / "SKILL.md"
        original = skill_file.read_text(encoding="utf-8")
        proposal_body = path.read_text(encoding="utf-8")
        skill_file.write_text(
            original + f"\n\n## 自进化补充({path.stem})\n{proposal_body}\n",
            encoding="utf-8")
        ok, output = gate.check()
        if ok:
            release.promote(skill_name, skill_file.read_text(encoding="utf-8"),
                            note=f"自进化:{path.stem}")
            print(f"[3/4] 门禁通过,{skill_name} 已灰度上线")
        else:
            skill_file.write_text(original, encoding="utf-8")
            print(f"[3/4] 门禁拦截,{skill_name} 已自动回滚\n{output}")
    print("[4/4] 闭环结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
