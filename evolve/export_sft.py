"""SFT 数据导出:从 traces 提炼优质会话为 OpenAI messages 格式 JSONL。

口径:"成功会话" = 无 handoff、无 guardrail_denied;
PII 脱敏(手机号/身份证/邮箱)在导出前强制执行。
真实训练命令见 README(本模块只做数据管线与格式校验)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PII_PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), "[PHONE]"),
    (re.compile(r"\d{17}[\dXx]"), "[ID]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
]


def mask_pii(text: str) -> str:
    for pattern, repl in PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def load_sessions(trace_dir: str) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for path in sorted(Path(trace_dir).glob("session-*.jsonl")):
        spans = [json.loads(line) for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sid = spans[0].get("session_id", path.stem) if spans else path.stem
        sessions[sid] = spans
    return sessions


def is_successful(spans: list[dict]) -> bool:
    bad = {"handoff", "guardrail_denied"}
    return not any(s.get("span") in bad for s in spans)


def session_to_samples(spans: list[dict]) -> list[dict]:
    """把 user_message → assistant_reply 配对为 SFT 样本(含工具调用回合的
    tool_result 上下文拼接进 assistant 回复前的 user 侧,保持因果可学)。"""
    samples = []
    pending_user: str | None = None
    for span in spans:
        name = span.get("span")
        if name == "user_message":
            pending_user = span.get("content", "")
        elif name == "assistant_reply" and pending_user:
            samples.append({"messages": [
                {"role": "user", "content": mask_pii(pending_user)},
                {"role": "assistant",
                 "content": mask_pii(span.get("content", ""))},
            ]})
            pending_user = None
    return samples


def export(trace_dir: str = "traces",
           out_file: str = "evolve/out/sft.jsonl") -> int:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for spans in load_sessions(trace_dir).values():
            if not is_successful(spans):
                continue
            for sample in session_to_samples(spans):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
    return count


def validate(out_file: str) -> list[str]:
    """JSONL schema 校验 + PII 泄漏扫描。"""
    errors = []
    for i, line in enumerate(Path(out_file).read_text(encoding="utf-8")
                             .splitlines()):
        try:
            sample = json.loads(line)
            messages = sample["messages"]
            assert messages[0]["role"] == "user"
            assert messages[-1]["role"] == "assistant"
        except (json.JSONDecodeError, KeyError, AssertionError,
                IndexError) as exc:
            errors.append(f"第{i + 1}行格式错误: {exc}")
            continue
        joined = json.dumps(sample, ensure_ascii=False)
        for pattern, _ in PII_PATTERNS:
            if pattern.search(joined):
                errors.append(f"第{i + 1}行存在未脱敏 PII: {pattern.pattern}")
    return errors


def main() -> None:
    import sys
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "evolve/out/sft.jsonl"
    count = export(trace_dir, out)
    errors = validate(out)
    print(f"导出 {count} 条 SFT 样本 → {out}")
    if errors:
        print("校验失败:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("schema 校验 + PII 扫描通过。")


if __name__ == "__main__":
    main()
