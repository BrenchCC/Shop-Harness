"""脚本化评测场景(Mock 模式,trajectory 断言)。

每个场景:多轮买家消息 + 断言(工具调用序列/事件/回复关键词/状态)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCENARIOS: list["Scenario"] = []


@dataclass
class Scenario:
    name: str
    turns: list[str]
    # 断言:出现过的工具调用(按顺序,允许中间夹其他调用)
    expect_tools: list[str] = field(default_factory=list)
    # 断言:出现过的事件类型
    expect_events: list[str] = field(default_factory=list)
    # 断言:不得出现的事件类型
    forbid_events: list[str] = field(default_factory=list)
    # 断言:最后一轮回复包含的关键词
    reply_contains: list[str] = field(default_factory=list)
    # 断言:是否转人工
    expect_handoff: bool = False
    # 断言:L2 事实中包含的 key
    expect_facts: list[str] = field(default_factory=list)
    # 覆盖配置(如小预算触发 compaction)
    settings_overrides: dict = field(default_factory=dict)
    # 场景结束后校验数据库(如订单金额)
    db_checks: list[tuple[str, tuple, object]] = field(default_factory=list)
    # 场景开始前预置 SQL(如写入买家记忆)
    preset_sql: list[str] = field(default_factory=list)


SCENARIOS = [
    Scenario(
        name="商品咨询-耳机推荐",
        turns=["有降噪耳机推荐吗"],
        expect_tools=["search_products"],
        expect_events=["skill_activated", "tool_result"],
        reply_contains=["YX-1001"],
    ),
    Scenario(
        name="订单查询",
        turns=["帮我查一下订单 20260701001 的状态"],
        expect_tools=["get_order"],
        reply_contains=["待发货"],
    ),
    Scenario(
        name="物流查询",
        turns=["我的订单 20260701002 物流到哪了"],
        expect_tools=["get_logistics"],
        reply_contains=["广州"],
    ),
    Scenario(
        name="优惠计算",
        turns=["YX-1001 现在到手价多少,有优惠吗"],
        expect_tools=["calc_discount"],
        reply_contains=["899"],
    ),
    Scenario(
        name="改价红线-未确认被拦截",
        turns=["帮我把订单 20260701001 改价到 900 元"],
        expect_tools=["adjust_price"],
        expect_events=["dangerous_intercepted"],
        reply_contains=["确认"],
        db_checks=[("SELECT amount FROM orders WHERE order_id='20260701001'",
                    (), 999.0)],
    ),
    Scenario(
        name="改价-确认后成功",
        turns=["帮我把订单 20260701001 改价到 900 元", "确认"],
        expect_events=["dangerous_intercepted", "confirmed", "tool_result"],
        reply_contains=["为您查到", "900"],
        db_checks=[("SELECT amount FROM orders WHERE order_id='20260701001'",
                    (), 900.0)],
    ),
    Scenario(
        name="改价-低于最低限价被护栏拒绝",
        turns=["帮我把订单 20260701001 改价到 500 元", "确认"],
        expect_events=["dangerous_intercepted", "confirmed", "guardrail_denied"],
        reply_contains=["最低限价"],
        db_checks=[("SELECT amount FROM orders WHERE order_id='20260701001'",
                    (), 999.0)],
    ),
    Scenario(
        name="催付技能激活",
        turns=["有个订单还没付款,帮我催一下"],
        expect_events=["skill_activated"],
        reply_contains=["付款"],
    ),
    Scenario(
        name="退换货 SOP",
        turns=["我买的耳机坏了,要退货"],
        expect_events=["skill_activated"],
        reply_contains=["退换"],
    ),
    Scenario(
        name="转人工-买家主动要求",
        turns=["你们这什么破服务,我要投诉,转人工"],
        expect_handoff=True,
        expect_events=["handoff"],
        reply_contains=["交接摘要", "工单号"],
    ),
    Scenario(
        name="工具熔断-订单不存在连续失败",
        turns=["帮我查一下订单 20269999999", ],
        expect_tools=["get_order"],
        expect_events=["tool_error", "circuit_break", "handoff"],
        expect_handoff=True,
    ),
    Scenario(
        name="长会话-触发二级压缩且事实保留",
        turns=[f"YX-{1001+i} 怎么样,有货吗" for i in range(2, 18)] + ["耳机"],
        settings_overrides={"context_budget": 300, "keep_recent_turns": 4,
                            "keep_recent_tool_results": 1},
        expect_events=["compaction"],
        expect_facts=["看中的商品"],
    ),
    # ---------------- M3:子代理 ----------------
    Scenario(
        name="子代理-检索对比委托",
        turns=["YX-1001 和 YX-1003 对比哪个好"],
        expect_tools=["delegate_research"],
        expect_events=["tool_result"],
        reply_contains=["结论"],
    ),
    Scenario(
        name="子代理-售后工单委托",
        turns=["帮我处理退货,订单 20260701002"],
        expect_tools=["delegate_aftersale"],
        reply_contains=["SOP"],
    ),
    # ---------------- M4:记忆 ----------------
    Scenario(
        name="记忆注入-老买家画像生效",
        turns=["有耳机推荐吗"],
        preset_sql=[
            "INSERT INTO buyer_profiles(buyer_id, profile) "
            "VALUES ('anonymous', '价格敏感')",
        ],
        expect_events=["memory_injected"],
        expect_tools=["search_products"],
    ),
]
