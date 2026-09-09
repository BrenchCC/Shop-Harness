"""SQLite 建库与种子数据。

50 件虚构商品的种子数据围绕 demo/eval 场景设计:
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
    created_at TEXT NOT NULL,
    buyer_id TEXT NOT NULL DEFAULT ''
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
    ("YX-1004", "音弦便携蓝牙音箱", "数码影音", 189.0, 159.0, 180,
     "双扬声器;续航12小时;附便携挂绳;适合桌面与露营"),
    ("YX-1005", "音弦领夹无线麦克风", "数码影音", 239.0, 199.0, 120,
     "一拖二收音;USB-C接收器;充电盒收纳;适合访谈录音"),
    ("YX-1006", "音弦桌面手机支架", "数码影音", 49.0, 35.0, 360,
     "角度可调;折叠收纳;防滑底座;支持横竖屏摆放"),
    ("YX-2003", "极光USB-C扩展坞", "电脑外设", 159.0, 129.0, 200,
     "HDMI与USB接口;支持PD供电;铝合金外壳;外接显示器需设备支持视频输出"),
    ("YX-2004", "极光笔记本电脑支架", "电脑外设", 89.0, 69.0, 230,
     "六档高度调节;铝合金材质;折叠便携;适合13至16英寸笔记本"),
    ("YX-2005", "极光加大桌面鼠标垫", "电脑外设", 39.0, 29.0, 510,
     "800×300mm;织物表面;橡胶防滑底;包边设计"),
    ("YX-2006", "极光1080P网络摄像头", "电脑外设", 199.0, 169.0, 140,
     "USB即插即用;内置麦克风;隐私遮挡盖;适合线上会议"),
    ("YX-3003", "云朵遮光窗帘 单片", "家居生活", 129.0, 99.0, 190,
     "宽1.5米高2.7米;挂钩安装;可机洗;下单前请测量窗户尺寸"),
    ("YX-3004", "云朵可折叠收纳箱 40L", "家居生活", 59.0, 45.0, 330,
     "透明翻盖;可叠放;折叠节省空间;适合衣物与玩具收纳"),
    ("YX-3005", "云朵浴室防滑地垫", "家居生活", 45.0, 32.0, 260,
     "40×60cm;吸水绒面;防滑底层;可清洗晾干"),
    ("YX-3006", "云朵双层玻璃水杯 350ml", "家居生活", 69.0, 49.0, 170,
     "双层隔热;带杯盖;透明杯身;不可用于明火加热"),
    ("YX-4003", "山野轻量徒步鞋", "服饰鞋包", 329.0, 279.0, 150,
     "橡胶防滑鞋底;透气鞋面;缓震鞋垫;尺码36至44"),
    ("YX-4004", "山野通勤双肩背包 20L", "服饰鞋包", 199.0, 159.0, 210,
     "独立电脑夹层;双侧水杯袋;加宽肩带;适合15.6英寸笔记本"),
    ("YX-4005", "山野折叠遮阳帽", "服饰鞋包", 69.0, 49.0, 300,
     "宽帽檐;可调节围度;可拆防风绳;折叠便携"),
    ("YX-4006", "山野棉质运动袜 5双装", "服饰鞋包", 49.0, 35.0, 420,
     "棉混纺面料;毛圈袜底;中筒设计;黑白灰基础配色"),
    ("YX-5003", "小魔方LED阅读台灯", "生活电器", 139.0, 109.0, 240,
     "三档色温;亮度可调;柔光灯罩;USB-C供电"),
    ("YX-5004", "小魔方便携电动打蛋器", "生活电器", 99.0, 79.0, 180,
     "五档速度;双搅拌棒;一键退棒;配件可拆洗"),
    ("YX-5005", "小魔方桌面循环风扇", "生活电器", 169.0, 139.0, 210,
     "三档风速;上下角度可调;定时关闭;前网罩可拆洗"),
    ("YX-6003", "元气混合坚果 7袋装", "食品酒水", 49.9, 39.9, 540,
     "每日独立小包装;含腰果杏仁核桃;含坚果过敏原;开封即食"),
    ("YX-6004", "元气原味燕麦片 1kg", "食品酒水", 35.9, 28.9, 460,
     "配料为燕麦;热水冲泡;拉链袋包装;适合早餐搭配"),
    ("YX-6005", "元气茉莉花茶 20袋装", "食品酒水", 39.9, 29.9, 380,
     "独立茶包;茉莉花香;冷热泡皆可;茶叶含天然咖啡因"),
    ("YX-7003", "乐读木质拼图 60片", "母婴玩具", 59.0, 45.0, 220,
     "海洋动物主题;圆角拼片;适合4岁以上;需成人陪同使用"),
    ("YX-7004", "乐读大颗粒积木 80粒", "母婴玩具", 129.0, 99.0, 160,
     "大颗粒易抓握;附收纳盒;多色搭配;适合3岁以上"),
    ("YX-7005", "乐读儿童涂鸦画板", "母婴玩具", 89.0, 69.0, 200,
     "双面书写;附水性画笔;可擦写;适合3岁以上"),
    ("YX-8003", "轻氧弹力训练带 3条装", "运动户外", 59.0, 45.0, 310,
     "三档阻力;附收纳袋;适合热身与拉伸;使用前检查有无破损"),
    ("YX-8004", "轻氧不锈钢保温运动水壶 600ml", "运动户外", 109.0, 85.0, 250,
     "304不锈钢内胆;防漏杯盖;提绳设计;不适合盛装碳酸饮料"),
    ("YX-8005", "轻氧折叠露营椅", "运动户外", 159.0, 129.0, 130,
     "钢管支架;侧边收纳袋;附手提袋;建议承重不超过100kg"),
    ("YX-9003", "素颜保湿身体乳 300ml", "美妆个护", 79.0, 59.0, 320,
     "乳液质地;含甘油;泵头取用;初次使用建议局部试用"),
    ("YX-9004", "素颜柔软洁面巾 80抽", "美妆个护", 29.9, 22.9, 680,
     "干湿两用;抽取式包装;无香型;不可直接冲入马桶"),
    ("YX-9005", "素颜旅行分装瓶 6件套", "美妆个护", 39.0, 29.0, 290,
     "含按压瓶与喷雾瓶;附标签贴;透明收纳袋;勿装高浓度酒精"),
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
    """Initialize db_path and add missing seed products without overwriting existing data."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(orders)")}
    legacy_orders = "buyer_id" not in columns
    if legacy_orders:
        conn.execute("ALTER TABLE orders ADD COLUMN buyer_id TEXT NOT NULL DEFAULT ''")
    fresh = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0
    # 已有数据库只补商品，不覆盖运营修改 / Add missing products, preserve edits.
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?)", PRODUCTS,
        )
        if fresh:
            conn.executemany(
                "INSERT OR IGNORE INTO orders "
                "(order_id, sku, quantity, amount, status, buyer, address, note, created_at, buyer_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(*order, "buyer-demo") for order in ORDERS],
            )
            conn.executemany("INSERT OR IGNORE INTO logistics VALUES (?,?,?)", LOGISTICS)
            conn.executemany(
                "INSERT INTO coupons(sku, threshold, discount) VALUES (?,?,?)", COUPONS,
            )
            conn.executemany(
                "INSERT INTO faqs(question, answer, category) VALUES (?,?,?)", FAQS,
            )
        if legacy_orders:
            # 只关联已知演示订单，未知历史订单保持未关联 / Migrate demo ownership only.
            conn.executemany(
                "UPDATE orders SET buyer_id = 'buyer-demo' "
                "WHERE order_id = ? AND buyer = ? AND buyer_id = ''",
                [(order[0], order[5]) for order in ORDERS],
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_buyer ON orders(buyer_id)")
    return conn
