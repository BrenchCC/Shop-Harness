"""自进化闭环与数据飞轮:bad case 挖掘、提案、门禁、灰度回滚、SFT/DPO 导出。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolve import export_dpo, export_sft, mine_bad_cases, propose  # noqa: E402
from evolve.release import SkillRelease  # noqa: E402
from shopharness.llm.mock_client import MockLLM  # noqa: E402


def write_trace(trace_dir: Path, session: str, spans: list[dict]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / f"session-{session}.jsonl").open("w",
                                                       encoding="utf-8") as f:
        for span in spans:
            span.setdefault("session_id", session)
            f.write(json.dumps(span, ensure_ascii=False) + "\n")


@pytest.fixture()
def trace_dir(tmp_path):
    d = tmp_path / "traces"
    # 成功会话(含 PII,检验脱敏)
    write_trace(d, "ok-1", [
        {"span": "user_message", "content": "耳机推荐,我电话13812345678"},
        {"span": "assistant_reply", "content": "推荐 YX-1001 降噪耳机"},
    ])
    # 转人工会话(与成功会话同 prompt,构成 DPO 对)
    write_trace(d, "bad-1", [
        {"span": "user_message", "content": "耳机推荐,我电话13812345678"},
        {"span": "assistant_reply", "content": "不知道,你自己看吧"},
        {"span": "handoff", "reason": "买家主动要求(投诉)"},
    ])
    # 护栏拒绝 + 多次工具错误会话
    write_trace(d, "bad-2", [
        {"span": "guardrail_denied", "reason": "低于最低限价"},
        {"span": "tool_error", "error": "x"},
        {"span": "tool_error", "error": "y"},
    ])
    return d


def test_mine_bad_cases(trace_dir):
    cases = mine_bad_cases.mine(str(trace_dir))
    types = sorted(c.type for c in cases)
    assert "handoff" in types
    assert "guardrail_denied" in types
    assert "tool_errors" in types


def test_propose_with_mock(trace_dir, tmp_path):
    bad_file = tmp_path / "bad.json"
    cases = mine_bad_cases.mine(str(trace_dir))
    from dataclasses import asdict
    bad_file.write_text(json.dumps([asdict(c) for c in cases],
                                   ensure_ascii=False), encoding="utf-8")
    written = propose.propose_all(str(bad_file), "skills", MockLLM(),
                                  out_dir=str(tmp_path / "proposals"))
    assert written
    content = written[0].read_text(encoding="utf-8")
    assert "提案" in content and "建议修改点" in content


def test_release_promote_and_rollback(tmp_path, settings):
    skills_dir = tmp_path / "skills"
    shutil.copytree("skills", skills_dir)
    rel = SkillRelease(settings.db_path, skills_dir=str(skills_dir),
                       versions_dir=str(tmp_path / "versions"))
    target = skills_dir / "inquiry-conversion" / "SKILL.md"
    original = target.read_text(encoding="utf-8")
    rel.promote("inquiry-conversion", original + "\n新增指令 V2\n")
    assert "V2" in target.read_text(encoding="utf-8")
    rel.rollback("inquiry-conversion")
    assert "V2" not in target.read_text(encoding="utf-8")
    history = rel.history("inquiry-conversion")
    assert len(history) >= 2
    assert history[0]["status"] == "rolled_back"


def test_export_sft_masks_pii(trace_dir, tmp_path):
    out = tmp_path / "sft.jsonl"
    count = export_sft.export(str(trace_dir), str(out))
    assert count >= 1
    content = out.read_text(encoding="utf-8")
    assert "13812345678" not in content
    assert "[PHONE]" in content
    assert export_sft.validate(str(out)) == []


def test_export_sft_skips_failed_sessions(trace_dir, tmp_path):
    out = tmp_path / "sft.jsonl"
    export_sft.export(str(trace_dir), str(out))
    content = out.read_text(encoding="utf-8")
    assert "你自己看吧" not in content  # handoff 会话不导出


def test_export_dpo_pairs(trace_dir, tmp_path):
    out = tmp_path / "dpo.jsonl"
    count = export_dpo.export(str(trace_dir), str(out))
    assert count == 1
    pair = json.loads(out.read_text(encoding="utf-8").strip())
    assert pair["chosen"] == "推荐 YX-1001 降噪耳机"
    assert pair["rejected"] == "不知道,你自己看吧"
    assert export_dpo.validate(str(out)) == []


def test_gate_script_runs():
    from evolve import gate
    ok, output = gate.check()
    assert ok, output
