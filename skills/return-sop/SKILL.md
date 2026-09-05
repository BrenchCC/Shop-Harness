---
name: return-sop
intents: 退货, 退款, 换货, 质量问题, 坏了, 售后
tools: get_order, get_logistics, search_faq, create_ticket, delegate_aftersale
description: 退换货标准作业流程
---
# 退换货 SOP

1. 用 get_order 核实订单:确认商品、金额、状态。
2. 判断售后期限:签收 7 天内支持无理由退换(定制/食品类除外)。
3. 质量问题:表达歉意,优先给换货方案;买家坚持退款则告知流程。
4. 需要人工审核的(金额 > 500、纠纷、超时售后):用 create_ticket 建工单,
   告知买家工单号与预计响应时间(24 小时内)。
5. 全程不承诺「立即退款到账」,统一话术为「原路退回,1-3 个工作日」。
