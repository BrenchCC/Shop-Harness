"""DPO 偏好对导出:相同买家输入下,被采纳回复(chosen) vs
触发转人工前的失败回复(rejected)。

配对规则:rejected = 转人工会话中最后一条 assistant 回复及其 user prompt;
chosen = 成功会话中相同 user prompt 的 assistant 回复。
"""

from __future__ import annotations

import json
from pathlib import Path

from evolve.export_sft import load_sessions, mask_pii


def extract_pairs(trace_dir: str) -> list[dict]:
    chosen_by_prompt: dict[str, str] = {}
    rejected: list[dict] = []
    for spans in load_sessions(trace_dir).values():
        last_user: str | None = None
        last_assistant: str | None = None
        has_handoff = any(s.get("span") == "handoff" for s in spans)
        for span in spans:
            if span.get("span") == "user_message":
                last_user = span.get("content", "")
            elif span.get("span") == "assistant_reply":
                last_assistant = span.get("content", "")
                if not has_handoff and last_user:
                    chosen_by_prompt[mask_pii(last_user)] = \
                        mask_pii(last_assistant)
        if has_handoff and last_user and last_assistant:
            rejected.append({"prompt": mask_pii(last_user),
                             "rejected": mask_pii(last_assistant)})
    pairs = []
    for item in rejected:
        if chosen := chosen_by_prompt.get(item["prompt"]):
            pairs.append({"prompt": item["prompt"], "chosen": chosen,
                          "rejected": item["rejected"]})
    return pairs


def export(trace_dir: str = "traces",
           out_file: str = "evolve/out/dpo.jsonl") -> int:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    pairs = extract_pairs(trace_dir)
    with open(out_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)


def validate(out_file: str) -> list[str]:
    errors = []
    for i, line in enumerate(Path(out_file).read_text(encoding="utf-8")
                             .splitlines()):
        try:
            pair = json.loads(line)
            assert pair["prompt"] and pair["chosen"] and pair["rejected"]
            assert pair["chosen"] != pair["rejected"]
        except (json.JSONDecodeError, KeyError, AssertionError) as exc:
            errors.append(f"第{i + 1}行格式错误: {exc}")
    return errors


def main() -> None:
    import sys
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "evolve/out/dpo.jsonl"
    count = export(trace_dir, out)
    errors = validate(out)
    print(f"导出 {count} 条 DPO 偏好对 → {out}")
    if errors:
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("schema 校验通过。")


if __name__ == "__main__":
    main()
