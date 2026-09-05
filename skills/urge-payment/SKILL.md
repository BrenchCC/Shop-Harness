---
name: urge-payment
intents: 催付, 还没付款, 未付款, 待付款, 忘了付
tools: get_order, calc_discount
description: 催付跟进 SOP,针对待付款订单
---
# 催付跟进 SOP

1. 先用 get_order 确认订单确实为「待付款」状态。
2. 语气友好不施压:先询问是否遇到问题(价格、地址、支付方式)。
3. 告知订单保留时限,提示库存情况(以工具数据为准)。
4. 如有可用优惠,用 calc_discount 算出到手价一并告知。
5. 买家明确表示不买了:礼貌结束,可用 add_order_note 记录原因。
