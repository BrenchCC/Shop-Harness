"""技能灰度发布:版本快照、promote、rollback。

技能 = 程序性记忆,版本化是"自进化可回滚"的前提。
版本文件存 evolve/versions/<skill>/<version>.md,元数据落 skill_versions 表。
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from shopharness.core.memory import SCHEMA as MEMORY_SCHEMA


class SkillRelease:
    def __init__(self, db_path: str, skills_dir: str = "skills",
                 versions_dir: str = "evolve/versions"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(MEMORY_SCHEMA)
        self.skills_dir = Path(skills_dir)
        self.versions_dir = Path(versions_dir)

    def _skill_file(self, skill_name: str) -> Path:
        return self.skills_dir / skill_name / "SKILL.md"

    def _next_version(self, skill_name: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM skill_versions WHERE skill_name = ?",
            (skill_name,)).fetchone()
        return (row["v"] or 0) + 1

    def snapshot(self, skill_name: str, status: str = "active",
                 note: str = "") -> int:
        """把当前技能文件存为版本,返回版本号。"""
        version = self._next_version(skill_name)
        dest_dir = self.versions_dir / skill_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{version}.md"
        shutil.copy(self._skill_file(skill_name), dest)
        self.conn.execute(
            "INSERT INTO skill_versions(skill_name, version, path, status, note)"
            " VALUES (?,?,?,?,?)",
            (skill_name, version, str(dest), status, note))
        self.conn.commit()
        return version

    def promote(self, skill_name: str, new_content: str,
                note: str = "") -> int:
        """灰度上线:先快照当前版本(回滚点),再写入新内容。"""
        self.snapshot(skill_name, status="active", note="promote 前自动快照")
        self._skill_file(skill_name).write_text(new_content, encoding="utf-8")
        return self.snapshot(skill_name, status="active",
                             note=note or "promote")

    def rollback(self, skill_name: str, note: str = "") -> int:
        """回滚到上一个 active 版本。"""
        rows = self.conn.execute(
            "SELECT * FROM skill_versions WHERE skill_name = ? "
            "ORDER BY version DESC", (skill_name,)).fetchall()
        if len(rows) < 2:
            raise RuntimeError(f"{skill_name} 没有可回滚的历史版本")
        previous = rows[1]
        shutil.copy(previous["path"], self._skill_file(skill_name))
        self.conn.execute(
            "UPDATE skill_versions SET status = 'rolled_back' "
            "WHERE skill_name = ? AND version = ?",
            (skill_name, rows[0]["version"]))
        self.conn.commit()
        return previous["version"]

    def history(self, skill_name: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT version, status, note, created_at FROM skill_versions "
            "WHERE skill_name = ? ORDER BY version DESC",
            (skill_name,)).fetchall()
        return [dict(r) for r in rows]


def main() -> None:
    import sys
    if len(sys.argv) < 3:
        print("用法: release.py <db_path> <promote|rollback|history> "
              "<skill_name> [提案文件]")
        sys.exit(1)
    db, cmd = sys.argv[1], sys.argv[2]
    rel = SkillRelease(db)
    if cmd == "history":
        for row in rel.history(sys.argv[3]):
            print(row)
    elif cmd == "promote":
        content = Path(sys.argv[4]).read_text(encoding="utf-8")
        print(f"已上线版本 {rel.promote(sys.argv[3], content)}")
    elif cmd == "rollback":
        print(f"已回滚到版本 {rel.rollback(sys.argv[3])}")


if __name__ == "__main__":
    main()
