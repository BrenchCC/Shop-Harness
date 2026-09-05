"""pytest 公共 fixture:内存级 tmp 数据库 + 可注入 Mock 的 Harness。"""

from __future__ import annotations

import pytest

from shopharness.cli import build_harness
from shopharness.config import Settings
from shopharness.llm.mock_client import MockLLM


@pytest.fixture()
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "shop.db"),
                    trace_dir=str(tmp_path / "traces"),
                    rag_enabled=False)  # 测试默认关向量检索,保持快速


@pytest.fixture()
def mock():
    return MockLLM()


@pytest.fixture()
def harness(settings, mock):
    return build_harness(settings, mock)


def event_types(result):
    return [e.type for e in result.events]


def tool_calls(result):
    return [e.detail.split("(")[0] for e in result.events
            if e.type == "tool_call"]
