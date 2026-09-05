"""SQLite 建库与种子数据。

种子数据围绕 demo/eval 场景设计:
- YX-1001 无线降噪耳机(改价剧情主角:售价 999,最低限价 880)
- 订单 20260701001(YX-1001,待发货)/ 20260701002(已发货有物流)/ 20260701003(待付款,催付场景)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    min_price REAL NOT NULL,
    stock INTEGER NOT NULL,
    selling_points TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL REFERENCES products(sku),
    quantity INTEGER NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,           -- 待付款/待发货/已发货/已完成
    buyer TEXT NOT NULL,
    address TEXT NOT NULL,
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logistics (
    order_id TEXT PRIMARY KEY REFERENCES orders(order_id),
    status TEXT NOT NULL,
    trace TEXT NOT NULL             -- JSON 数组字符串
);
CREATE TABLE IF NOT EXISTS coupons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL REFERENCES products(sku),
    threshold REAL NOT NULL,        -- 满 threshold 减 discount
    discount REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    args TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS faqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category TEXT NOT NULL
);
"""

PRODUCTS = [
    ("YX-1001", "音弦无线降噪耳机 Pro", "数码影音", 999.0, 880.0, 156,
     "主动降噪45dB;续航40小时;蓝牙5.4;支持双设备连接;一年只换不修"),
    ("YX-1002", "音弦半入耳蓝牙耳机 Air", "数码影音", 299.0, 259.0, 482,
     "单耳3.8g轻若无物;通话降噪;续航28小时;开盖即连"),
    ("YX-1003", "音弦头戴式监听耳机 Studio", "数码影音", 1599.0, 1399.0, 37,
     "50mm动圈;Hi-Res金标认证;可换线设计;附赠便携收纳包"),
    ("YX-2001", "极光机械键盘 87键", "电脑外设", 399.0, 349.0, 210,
     "Gasket结构;三模连接;热插拔轴座;PBT键帽"),
    ("YX-2002", "极光静音办公鼠标", "电脑外设", 129.0, 99.0, 660,
     "静音微动;2.4G+蓝牙双模;人体工学;一节电池用一年"),
    ("YX-3001", "云朵记忆棉枕头", "家居生活", 169.0, 139.0, 320,
     "泰国进口乳胶;波浪曲线护颈;可拆洗枕套;30天试睡"),
    ("YX-3002", "云朵全棉四件套 1.8m床", "家居生活", 329.0, 289.0, 145,
     "100支长绒棉;A类母婴级面料;活性印染不褪色"),
    ("YX-4001", "山野冲锋衣 三合一", "服饰鞋包", 699.0, 599.0, 88,
     "暴雨级防水;抓绒内胆可拆;YKK拉链;终身保修"),
    ("YX-4002", "山野速干T恤", "服饰鞋包", 99.0, 79.0, 530,
     "Coolmax速干面料;UPF50+防晒;抑菌防臭"),
    ("YX-5001", "小魔方迷你加湿器", "生活电器", 89.0, 69.0, 410,
     "300ml水箱;静音≤30dB;无水自动断电;七彩夜灯"),
    ("YX-5002", "小魔方桌面空气净化器", "生活电器", 459.0, 399.0, 96,
     "HEPA13滤网;除甲醛;PM2.5数显;静音睡眠模式"),
    ("YX-6001", "元气无糖气泡水 15瓶装", "食品酒水", 59.9, 49.9, 999,
     "0糖0脂0卡;白桃味;气泡绵密"),
    ("YX-6002", "元气冷萃咖啡液 10条装", "食品酒水", 79.0, 65.0, 720,
     "100%阿拉比卡;0蔗糖;冷水速溶;便携条装"),
    ("YX-7001", "乐读儿童绘本套装 12册", "母婴玩具", 139.0, 119.0, 260,
     "3-6岁情商启蒙;大豆油墨印刷;圆角设计防划伤"),
    ("YX-7002", "乐读点读笔", "母婴玩具", 269.0, 229.0, 175,
     "支持2000+绘本;中英双语;32G存储;防摔设计"),
    ("YX-8001", "轻氧瑜伽垫 6mm", "运动户外", 119.0, 95.0, 340,
     "TPE环保材质;双面防滑;附背带;体位线辅助"),
    ("YX-8002", "轻氧跳绳 智能计数", "运动户外", 69.0, 55.0, 480,
     "高清屏计数;无绳两用;轴承顺滑不绕绳"),
    ("YX-9001", "素颜氨基酸洗面奶", "美妆个护", 89.0, 72.0, 390,
     "纯氨基酸表活;温和不紧绷;敏感肌可用;150ml大容量"),
    ("YX-9002", "素颜玻尿酸精华 30ml", "美妆个护", 199.0, 169.0, 210,
     "5重玻尿酸;补水锁水;无酒精无香精"),
    ("YX-9101", "暖冬恒温暖杯垫", "生活电器", 79.0, 62.0, 275,
     "55℃恒温;重力感应自动开关;防水面板"),
]

ORDERS = [
    ("20260701001", "YX-1001", 1, 999.0, "待发货", "张先生",
     "浙江省杭州市西湖区文三路 100 号", "", "2026-07-01 10:23:00"),
    ("20260701002", "YX-2001", 1, 399.0, "已发货", "李女士",
     "广东省深圳市南山区科技园南路 20 号", "", "2026-07-30 14:02:00"),
    ("20260701003", "YX-4001", 1, 699.0, "待付款", "王先生",
     "北京市朝阳区望京 SOHO T3", "", "2026-07-30 21:45:00"),
]

LOGISTICS = [
    ("20260701002", "运输中",
     '[{"time":"2026-07-06 09:12","desc":"深圳转运中心已发出"},'
     '{"time":"2026-07-06 20:40","desc":"到达广州分拨中心"},'
     '{"time":"2026-07-07 08:15","desc":"发往杭州途中"}]'),
]

COUPONS = [
    ("YX-1001", 999.0, 100.0),   # 耳机满999减100
    ("YX-1002", 299.0, 30.0),
    ("YX-2001", 399.0, 40.0),
    ("YX-6001", 50.0, 5.0),
    ("YX-9002", 199.0, 20.0),
]

FAQS = [
    ("支持七天无理由退货吗", "大部分商品支持签收后 7 天无理由退换(食品、定制类除外),"
     "退货运费由买家承担,质量问题运费我们承担。", "售后"),
    ("发货要多久", "工作日 16 点前付款的订单当天发货,其余次日发货;"
     "默认中通/圆通,新疆西藏时效顺延 3-5 天。", "物流"),
    ("可以开发票吗", "支持电子普通发票,下单时备注抬头或发货后联系客服补开,"
     "1-3 个工作日发送至预留邮箱。", "售后"),
    ("耳机保修多久", "数码类产品一年质保,耳机支持一年只换不修(非人为损坏),"
     "保修需保留订单号作为凭证。", "售后"),
    ("优惠券可以叠加吗", "满减券与店铺折扣可叠加,多张满减券不可叠加,"
     "系统自动选用最优组合。", "优惠"),
    ("怎么查询物流", "提供订单号即可查询实时物流;发货后也会有短信通知,"
     "签收前请检查外包装是否完好。", "物流"),
    ("商品是正品吗", "本店为品牌授权旗舰店,所有商品正品保障,支持专柜验货,"
     "假一赔十。", "售前"),
    ("尺码怎么选", "服饰类建议参考详情页尺码表,介于两码之间建议拍大一码;"
     "不合身支持 7 天内免费换码。", "售前"),
]


def ensure_db(db_path: str) -> sqlite3.Connection:
    """建库(幂等)并写入种子数据,返回连接。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)", PRODUCTS)
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", ORDERS)
        conn.executemany(
            "INSERT INTO logistics VALUES (?,?,?)", LOGISTICS)
        conn.executemany(
            "INSERT INTO coupons(sku, threshold, discount) VALUES (?,?,?)",
            COUPONS)
        conn.executemany(
            "INSERT INTO faqs(question, answer, category) VALUES (?,?,?)",
            FAQS)
        conn.commit()
    return conn
