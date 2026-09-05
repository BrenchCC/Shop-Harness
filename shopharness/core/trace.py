"""轻量 JSONL trace 记录器。

span 字段名对齐 OpenTelemetry GenAI Semantic Conventions,
后续可无缝替换为 OTel SDK + Langfuse 导出。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, trace_dir: str = "traces", session_id: str | None = None,
                 enabled: bool = True):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        self._path = Path(trace_dir) / f"session-{self.session_id}.jsonl"
        if enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def span(self, name: str, **attrs: Any) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.time(),
            "session_id": self.session_id,
            "span": name,
            "gen_ai.system": "shopharness",
            **attrs,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class NullTracer(Tracer):
    def __init__(self) -> None:
        super().__init__(enabled=False)
