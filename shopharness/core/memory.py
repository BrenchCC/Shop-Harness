"""分层记忆(对应 DESIGN.md §5.1)。

- 情景记忆:session_summaries —— 每次会话结束时的摘要,按买家沉淀
- 语义记忆:buyer_profiles —— 买家画像(偏好、价格敏感度、投诉史)
- 程序性记忆:skill_versions —— 技能版本表(evolve/release.py 使用)

会话开始:画像 + 最近摘要注入 L1;会话结束:LLM 摘要 + 规则蒸馏偏好落库。
"""

from __future__ import annotations

import sqlite3
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS buyer_profiles (
    buyer_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    buyer_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS skill_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- active / rolled_back
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------ 语义记忆

    def get_profile(self, buyer_id: str) -> str:
        row = self.conn.execute(
            "SELECT profile FROM buyer_profiles WHERE buyer_id = ?",
            (buyer_id,)).fetchone()
        return row[0] if row else ""

    def update_profile(self, buyer_id: str, new_facts: list[str]) -> None:
        """追加去重式更新画像(确定性合并,不覆盖旧事实)。"""
        existing = self.get_profile(buyer_id)
        lines = [l for l in existing.split(";") if l.strip()] if existing else []
        for fact in new_facts:
            if fact and fact not in lines:
                lines.append(fact)
        self.conn.execute(
            "INSERT INTO buyer_profiles(buyer_id, profile, updated_at) "
            "VALUES (?,?,datetime('now','localtime')) "
            "ON CONFLICT(buyer_id) DO UPDATE SET "
            "profile=excluded.profile, updated_at=excluded.updated_at",
            (buyer_id, ";".join(lines)))
        self.conn.commit()

    # ------------------------------------------------------------ 情景记忆

    def add_session_summary(self, buyer_id: str, summary: str) -> None:
        self.conn.execute(
            "INSERT INTO session_summaries(buyer_id, summary) VALUES (?,?)",
            (buyer_id, summary))
        self.conn.commit()

    def recent_summaries(self, buyer_id: str, limit: int = 3) -> list[str]:
        rows = self.conn.execute(
            "SELECT summary FROM session_summaries WHERE buyer_id = ? "
            "ORDER BY id DESC LIMIT ?", (buyer_id, limit)).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------ L1 注入

    def build_l1_context(self, buyer_id: str) -> str:
        """组装注入 L1 的记忆文本;无记忆时返回空串(不占 token)。"""
        parts = []
        if profile := self.get_profile(buyer_id):
            parts.append(f"买家画像:{profile}")
        if summaries := self.recent_summaries(buyer_id):
            parts.append("近期会话:" + " | ".join(summaries))
        return "\n".join(parts)

    # ------------------------------------------------------------ 偏好蒸馏

    def distill_preferences(self, facts: dict[str, str],
                            action_log: list[str],
                            handed_off: bool) -> list[str]:
        """从 L2 事实与操作日志蒸馏买家偏好(确定性规则)。"""
        prefs: list[str] = []
        text = " ".join(f"{k}:{v}" for k, v in facts.items())
        if any(kw in text for kw in ("价格", "优惠", "便宜", "到手")):
            prefs.append("价格敏感")
        if any("adjust_price" in a for a in action_log):
            prefs.append("有改价行为")
        if handed_off:
            prefs.append("有转人工历史")
        if skus_text := facts.get("看中的商品"):
            categories = self._categories_of(skus_text)
            if categories:
                prefs.append("关注品类:" + ",".join(sorted(categories)))
        return prefs

    def _categories_of(self, skus_text: str) -> set[str]:
        import re
        categories = set()
        for sku in re.findall(r"YX-\d{4}", skus_text):
            row = self.conn.execute(
                "SELECT category FROM products WHERE sku = ?", (sku,)).fetchone()
            if row:
                categories.add(row[0])
        return categories
