"""改进提案生成:对每类 bad case,结合相关技能文件,让 LLM 产出修改提案。

原则(对应 DESIGN.md §5.2):自动化只到"提案"为止,绝不直接改线上技能;
提案必须过离线评测门禁(gate.py)后才允许灰度。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from shopharness.llm.base import LLMClient, Message

# bad case 类型 → 最相关的技能
TYPE_TO_SKILL = {
    "handoff": "return-sop",
    "guardrail_denied": "inquiry-conversion",
    "circuit_break": "return-sop",
    "tool_errors": "inquiry-conversion",
}

PROPOSE_PROMPT = """你是客服 Agent 的技能优化师。以下是一类线上 bad case 与当前技能文件,
请输出改进提案:1) 问题归因(1-2 句);2) 建议修改点(具体到指令条目);
3) 修改后应追加/替换的指令文本。输出 Markdown,不要输出无关内容。"""


def propose_for_type(case_type: str, cases: list[dict],
                     skill_content: str, llm: LLMClient) -> str:
    digest = "\n".join(f"- [{c['session_id']}] {c['detail']}"
                       for c in cases[:10])
    messages = [
        Message.system(PROPOSE_PROMPT),
        Message.user(f"bad case 类型:{case_type}\n样本:\n{digest}\n\n"
                     f"当前技能文件:\n{skill_content}"),
    ]
    resp = llm.chat(messages)
    return resp.content or ""


def propose_all(bad_cases_file: str, skills_dir: str, llm: LLMClient,
                out_dir: str = "evolve/out/proposals") -> list[Path]:
    cases = json.loads(Path(bad_cases_file).read_text(encoding="utf-8"))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_type[c["type"]].append(c)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case_type, group in by_type.items():
        skill_name = TYPE_TO_SKILL.get(case_type)
        if not skill_name:
            continue
        skill_file = Path(skills_dir) / skill_name / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8") \
            if skill_file.exists() else "(技能文件缺失)"
        proposal = propose_for_type(case_type, group, content, llm)
        path = out / f"{case_type}-{skill_name}.md"
        path.write_text(f"# 提案:{case_type} → {skill_name}\n\n{proposal}",
                        encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    import sys
    from shopharness.llm.mock_client import MockLLM
    bad_cases = sys.argv[1] if len(sys.argv) > 1 else "evolve/out/bad_cases.json"
    written = propose_all(bad_cases, "skills", MockLLM())
    for p in written:
        print(f"提案已生成:{p}")


if __name__ == "__main__":
    main()
