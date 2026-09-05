"""脚本化 Mock LLM:无 GPU 环境下让 pytest / eval / CLI demo 全功能可跑。

行为规则(按最新消息类型分派):
- system(内部调用):事实抽取 / 会话摘要,给出确定性输出
- tool:拦截→复述确认;护栏拒绝→致歉;错误→重试一次;成功→组织回复
- user:关键词路由到工具调用或话术回复

支持 queue 注入自定义响应序列(测试用,优先于规则)。
"""

from __future__ import annotations

import re

from ..llm.base import Message, ToolCall

ORDER_RE = re.compile(r"20\d{9}")
SKU_RE = re.compile(r"YX-\d{4}")
PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*元")

PRODUCT_WORDS = ["耳机", "键盘", "鼠标", "枕头", "四件套", "冲锋衣", "T恤",
                 "加湿器", "净化器", "气泡水", "咖啡", "绘本", "点读笔",
                 "瑜伽垫", "跳绳", "洗面奶", "精华", "杯垫"]


class MockLLM:
    def __init__(self, queue: list[Message] | None = None):
        self.queue: list[Message] = list(queue or [])
        self._seq = 0

    def chat(self, messages: list[Message],
             tools: list[dict] | None = None) -> Message:
        if self.queue:
            return self.queue.pop(0)
        system = messages[0].content or "" if messages else ""
        # ---- harness 内部调用:事实抽取 / 摘要
        if "抽取关键事实" in system:
            return Message.assistant(self._extract(messages))
        if "压缩为 100 字" in system:
            return Message.assistant(self._summarize(messages))
        if "技能优化师" in system:
            return Message.assistant(
                "## 问题归因\n技能指令对异常分支覆盖不足,模型在边界场景下行为发散。\n\n"
                "## 建议修改点\n- 增加一条明确的降级话术指令\n- 补充 bad case 对应的处理示例\n\n"
                "## 追加指令\n遇到无法处理的情况时,先安抚买家情绪,再给出一个可执行的"
                "下一步(换关键词重试/转人工),禁止沉默或重复提问。")
        # ---- 子代理模式:按子代理 system prompt 分派,避免嵌套委托
        if "检索子代理" in system:
            return self._research_agent(messages[-1])
        if "售后工单子代理" in system:
            return self._aftersale_agent(messages[-1])
        last = messages[-1]
        if last.role == "tool":
            return self._on_tool_result(messages, last)
        if last.role == "system" and "已明确确认" in (last.content or ""):
            return self._reissue_dangerous(messages)
        if last.role == "user":
            return self._route(last.content or "")
        return Message.assistant("您好,请问有什么可以帮您?")

    # ------------------------------------------------------------ 内部调用

    def _extract(self, messages: list[Message]) -> str:
        text = "\n".join(m.content or "" for m in messages)
        lines = []
        if skus := sorted(set(SKU_RE.findall(text))):
            lines.append("看中的商品: " + ", ".join(skus))
        if orders := sorted(set(ORDER_RE.findall(text))):
            lines.append("涉及的订单: " + ", ".join(orders))
        return "\n".join(lines) or "无"

    def _summarize(self, messages: list[Message]) -> str:
        text = "\n".join(m.content or "" for m in messages)
        skus = ",".join(sorted(set(SKU_RE.findall(text)))) or "无"
        orders = ",".join(sorted(set(ORDER_RE.findall(text)))) or "无"
        return (f"[Mock摘要] 压缩 {len(messages)} 条历史消息;"
                f"涉及商品:{skus};涉及订单:{orders}")

    # ------------------------------------------------------------ 工具结果

    def _on_tool_result(self, messages: list[Message], last: Message) -> Message:
        content = last.content or ""
        if "[系统拦截]" in content:
            call = self._last_tool_call(messages, "adjust_price")
            if call:
                args = call.arguments
                return Message.assistant(
                    f"确认一下:您希望将订单 {args.get('order_id')} 的金额改为 "
                    f"{args.get('new_price')} 元(原因:{args.get('reason', '买家要求')})。"
                    "确认后我将立即为您修改,请回复「确认」。")
            return Message.assistant("该操作需要您确认后才能执行,请回复「确认」。")
        if "[护栏拒绝]" in content:
            return Message.assistant(
                "非常抱歉,该价格低于店铺最低限价,我无权为您修改。"
                "您可以看看店内优惠券活动,或我为您申请赠品,您看可以吗?")
        if "[系统纠正]" in content:
            return Message.assistant("您好,请问有什么可以帮您?")
        if "[工具错误]" in content or '"error"' in content:
            # 软/硬失败:重试一次相同调用,再失败交给熔断
            same_errors = sum(
                1 for m in messages
                if m.role == "tool" and m.name == last.name
                and m.content and ('"error"' in m.content or "[工具错误]" in m.content))
            call = self._last_tool_call(messages, last.name or "")
            if same_errors <= 1 and call:
                return self._call(call.name, call.arguments)
            return Message.assistant(
                "很抱歉,暂时没有查到相关信息。您可以核对后重新提供,或我为您转人工处理。")
        # 成功结果 → 组织自然语言回复
        short = content if len(content) <= 300 else content[:300] + "…"
        return Message.assistant(f"为您查到:{short}。还有其他可以帮您的吗?")

    def _reissue_dangerous(self, messages: list[Message]) -> Message:
        call = self._last_tool_call(messages, "adjust_price")
        if call:
            return self._call("adjust_price", call.arguments)
        return Message.assistant("好的,已为您处理。")

    # ------------------------------------------------------------ 子代理模式

    def _research_agent(self, last: Message) -> Message:
        if last.role == "tool":
            content = (last.content or "")[:200]
            return Message.assistant(f"结论:{content}")
        text = last.content or ""
        word = next((w for w in PRODUCT_WORDS if w in text), None)
        return self._call("search_products", {"keyword": word or text[:20]})

    def _aftersale_agent(self, last: Message) -> Message:
        if last.role == "tool":
            content = (last.content or "")[:200]
            return Message.assistant(
                f"SOP 处理结论:已核实订单({content})。签收 7 天内可退换,"
                "建议优先换货;如需人工审核已可建工单。")
        text = last.content or ""
        order = ORDER_RE.search(text)
        return self._call("get_order",
                          {"order_id": order.group(0) if order else "20260701002"})

    # ------------------------------------------------------------ 用户路由

    def _route(self, text: str) -> Message:
        order_match = ORDER_RE.search(text)
        order_id = order_match.group(0) if order_match else None
        if any(kw in text for kw in ("对比", "比较", "哪个好", "怎么选")):
            return self._call("delegate_research", {"query": text})
        if "改价" in text:
            price = PRICE_RE.search(text)
            args = {"order_id": order_id or "20260701001",
                    "new_price": float(price.group(1)) if price else 900.0,
                    "reason": "买家要求改价"}
            return self._call("adjust_price", args)
        if any(kw in text for kw in ("物流", "到哪", "发货了吗")):
            return self._call("get_logistics",
                              {"order_id": order_id or "20260701002"})
        if any(kw in text for kw in ("催付", "还没付款", "未付款")):
            return Message.assistant(
                "亲,看到您的订单还未付款哦。请问是遇到什么问题了嘛?"
                "订单会为您保留 24 小时,喜欢的话可以尽快拍下,有满减活动哦。")
        if "处理" in text and any(kw in text for kw in ("退货", "退款", "换货")):
            return self._call("delegate_aftersale", {"issue": text})
        if "订单" in text:
            return self._call("get_order",
                              {"order_id": order_id or "20260701001"})
        if any(kw in text for kw in ("优惠", "便宜", "到手")):
            sku = SKU_RE.search(text)
            return self._call("calc_discount",
                              {"sku": sku.group(0) if sku else "YX-1001"})
        if any(kw in text for kw in ("退货", "退款", "换货", "售后")):
            return Message.assistant(
                "很抱歉给您带来不便。签收 7 天内支持无理由退换;"
                "如属质量问题我们优先为您换货。请提供订单号,我先为您核实订单状态。")
        word = next((w for w in PRODUCT_WORDS if w in text), None)
        if word or any(kw in text for kw in ("推荐", "怎么样", "好用吗")):
            return self._call("search_products", {"keyword": word or text})
        return Message.assistant("您好,请问有什么可以帮您?")

    # ------------------------------------------------------------ 工具

    def _call(self, name: str, arguments: dict) -> Message:
        self._seq += 1
        return Message.assistant(
            tool_calls=[ToolCall(id=f"mock-{self._seq}", name=name,
                                 arguments=arguments)])

    @staticmethod
    def _last_tool_call(messages: list[Message], name: str) -> ToolCall | None:
        for m in reversed(messages):
            if m.tool_calls:
                for call in m.tool_calls:
                    if call.name == name:
                        return call
        return None
