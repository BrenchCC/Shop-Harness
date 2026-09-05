"""Skills 系统:加载、路由、白名单裁剪、热加载。"""

from __future__ import annotations

from shopharness.core.skills import SkillManager, parse_skill_md


def manager() -> SkillManager:
    mgr = SkillManager(skills_dir="skills")
    mgr.load()
    return mgr


def test_load_builtin_skills():
    mgr = manager()
    assert set(mgr.skills) == {"inquiry-conversion", "urge-payment",
                               "return-sop"}
    skill = mgr.skills["inquiry-conversion"]
    assert "search_products" in skill.tools
    assert "转化" in skill.instructions


def test_route_activates_relevant_skill():
    mgr = manager()
    active = mgr.route("这款耳机怎么样,有优惠吗")
    assert active[0].name == "inquiry-conversion"
    assert mgr.route("订单还没付款")[0].name == "urge-payment"
    assert mgr.route("我要退货")[0].name == "return-sop"


def test_route_no_match():
    mgr = manager()
    assert mgr.route("asdfgh") == []


def test_tool_whitelist():
    mgr = manager()
    active = mgr.route("我要退货")
    whitelist = mgr.tool_whitelist(active)
    assert "get_order" in whitelist and "create_ticket" in whitelist
    assert "adjust_price" not in whitelist
    assert mgr.tool_whitelist([]) is None  # 无激活技能不裁剪


def test_max_two_active():
    mgr = manager()
    active = mgr.route("耳机优惠退货退款催付")  # 命中多个技能
    assert len(active) <= 2


def test_parse_skill_md_requires_frontmatter(tmp_path):
    bad = tmp_path / "SKILL.md"
    bad.write_text("没有 frontmatter", encoding="utf-8")
    try:
        parse_skill_md(bad)
        assert False, "应当抛 ValueError"
    except ValueError:
        pass


def test_hot_reload(tmp_path):
    skill_dir = tmp_path / "skills"
    (skill_dir / "s1").mkdir(parents=True)
    (skill_dir / "s1" / "SKILL.md").write_text(
        "---\nname: s1\nintents: 你好\ntools: get_order\ndescription: t\n---\n正文",
        encoding="utf-8")
    mgr = SkillManager(skills_dir=str(skill_dir))
    mgr.load()
    assert "s1" in mgr.skills
    (skill_dir / "s2").mkdir()
    (skill_dir / "s2" / "SKILL.md").write_text(
        "---\nname: s2\nintents: 再见\ntools: \ndescription: t\n---\n正文2",
        encoding="utf-8")
    mgr.reload()
    assert "s2" in mgr.skills
