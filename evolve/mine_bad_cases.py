"""bad case 挖掘:扫描 traces/*.jsonl,按规则归类问题会话。

规则(对应线上告警口径):
- handoff:发生转人工(含原因)
- guardrail_denied:护栏拒绝(可能是误拦或提示词诱导)
- circuit_break:工具熔断
- tool_errors:单会话工具错误 ≥ 2 次

输出 evolve/out/bad_cases.json,供 propose.py 消费。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BadCase:
    session_id: str
    type: str          # handoff / guardrail_denied / circuit_break / tool_errors
    detail: str
    trace_file: str


def mine_session(trace_file: Path) -> list[BadCase]:
    spans = []
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            spans.append(json.loads(line))
    session_id = spans[0].get("session_id", trace_file.stem) if spans \
        else trace_file.stem
    cases: list[BadCase] = []
    tool_errors = 0
    for span in spans:
        name = span.get("span")
        if name == "handoff":
            cases.append(BadCase(session_id, "handoff",
                                 span.get("reason", ""), str(trace_file)))
        elif name == "guardrail_denied":
            cases.append(BadCase(session_id, "guardrail_denied",
                                 span.get("reason", ""), str(trace_file)))
        elif name == "tool_error":
            tool_errors += 1
            if "circuit" in str(span.get("error", "")):
                pass
    if tool_errors >= 2:
        cases.append(BadCase(session_id, "tool_errors",
                             f"{tool_errors} 次工具错误", str(trace_file)))
    return cases


def mine(trace_dir: str = "traces") -> list[BadCase]:
    cases: list[BadCase] = []
    for path in sorted(Path(trace_dir).glob("session-*.jsonl")):
        cases.extend(mine_session(path))
    return cases


def main() -> None:
    import sys
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "traces"
    cases = mine(trace_dir)
    out = Path("evolve/out")
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "bad_cases.json"
    out_file.write_text(json.dumps([asdict(c) for c in cases],
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    by_type: dict[str, int] = {}
    for c in cases:
        by_type[c.type] = by_type.get(c.type, 0) + 1
    print(f"挖出 {len(cases)} 条 bad case:{by_type}")
    print(f"已写入 {out_file}")


if __name__ == "__main__":
    main()
