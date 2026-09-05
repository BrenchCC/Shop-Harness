"""Skills 系统:目录式技能(SKILL.md)、意图路由、热加载。

SKILL.md 格式:
    ---
    name: inquiry-conversion
    intents: 耳机, 推荐, 怎么样, 优惠
    tools: search_products, get_product_detail, calc_discount
    description: 询单转化话术
    ---
    (Markdown 指令正文,注入 L0 system)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    intents: list[str]
    tools: list[str]
    description: str
    instructions: str
    path: str = ""

    def score(self, text: str) -> int:
        return sum(text.count(kw) for kw in self.intents)


def parse_skill_md(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: 缺少 frontmatter")
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return Skill(
        name=meta["name"],
        intents=[s.strip() for s in meta.get("intents", "").split(",") if s.strip()],
        tools=[s.strip() for s in meta.get("tools", "").split(",") if s.strip()],
        description=meta.get("description", ""),
        instructions=body.strip(),
        path=str(path),
    )


@dataclass
class SkillManager:
    skills_dir: str
    skills: dict[str, Skill] = field(default_factory=dict)
    max_active: int = 2  # 同一轮最多激活的技能数,控制 prompt 体积

    def load(self) -> None:
        self.skills = {}
        base = Path(self.skills_dir)
        if not base.exists():
            return
        for path in sorted(base.glob("*/SKILL.md")):
            skill = parse_skill_md(path)
            self.skills[skill.name] = skill

    def reload(self) -> None:
        """热加载:技能文件变更后无需重启进程。"""
        self.load()

    def route(self, user_text: str) -> list[Skill]:
        """关键词打分路由,返回得分最高的 1-2 个技能。"""
        scored = [(s.score(user_text), s) for s in self.skills.values()]
        active = [s for score, s in sorted(scored, key=lambda x: -x[0])
                  if score > 0]
        return active[: self.max_active]

    def tool_whitelist(self, active: list[Skill]) -> set[str] | None:
        """无激活技能时不裁剪;有激活技能时取白名单并集 + create_ticket。"""
        if not active:
            return None
        whitelist: set[str] = {"create_ticket"}
        for skill in active:
            whitelist.update(skill.tools)
        return whitelist
