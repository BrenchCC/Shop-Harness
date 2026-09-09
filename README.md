# ShopHarness

面向电商客服场景的 **Agent Harness(脚手架)** —— 支持云端 API 与可选的本地 Qwen3-8B 部署，为模型提供客服工具、上下文和执行约束。
设计文档见 [DESIGN.md](DESIGN.md)(本仓库实现其 M1+M2 核心版)。

## 核心能力

| 模块 | 实现 | 位置 |
| --- | --- | --- |
| Harness 主 loop | turn 管理、最大步数、非法工具自我纠正、连续失败熔断 | `shopharness/core/harness.py` |
| 上下文工程 | 分层上下文(L0 技能指令 / L1 记忆 / L2 结构化状态 / L3+L4 历史)+ 三级 compaction | `shopharness/core/context.py` |
| 工具系统 | 10 个业务工具(SQLite),OpenAI function calling schema,按技能白名单动态裁剪 | `shopharness/tools/` |
| **RAG 检索增强** | 云端 Embedding 或本地 bge-small-zh + 关键词检索,商品 RRF 混合排序;商品库 + FAQ 知识库;向量不可用时自动降级 | `shopharness/core/rag.py` |
| 权限模型 | READ / WRITE / DANGEROUS 三级;改价须经买家复述确认 + 最低限价护栏 + 审计落库 | `shopharness/core/permissions.py`、`hooks.py` |
| Skills | 目录式 SKILL.md(询单转化 / 订单查询 / 催付 / 退换 SOP),意图路由激活,热加载 | `skills/`、`core/skills.py` |
| 转人工 | 关键词/熔断/步数超限触发,自动生成交接摘要并建工单 | `shopharness/core/handoff.py` |
| **子代理(M3)** | 上下文隔离的检索/售后子代理,仅回传结论摘要;注册为 `delegate_*` 工具 | `shopharness/core/subagent.py` |
| **长程流程(M3)** | LangGraph 售后工单流程,interrupt 等买家确认,SqliteSaver checkpoint 跨进程恢复 | `shopharness/flows/aftersale.py` |
| **分层记忆(M4)** | 情景(会话摘要)/ 语义(买家画像)/ 程序性(技能版本)三层,L1 注入 | `shopharness/core/memory.py` |
| **自进化(M4)** | bad case 挖掘 → LLM 提案 → 离线门禁 → 灰度/回滚,dry-run 默认 | `evolve/` |
| **数据飞轮(M4)** | traces → SFT/DPO JSONL 导出,PII 脱敏 + schema 校验 | `evolve/export_*.py` |
| 可观测性 | JSONL trace,字段对齐 OTel GenAI 语义约定 | `shopharness/core/trace.py` |
| 评测 | 18 条脚本化场景,trajectory 断言 + `--gate` 回归门禁 | `eval/` |

## 安装与 Mock 模式

核心项目支持 Python 3.10 及以上，无需 GPU。安装基础包和测试依赖不会安装 vLLM、PyTorch 或 Transformers。

```bash
pip install -e '.[dev]'

python -m pytest
python eval/run_eval.py --gate

printf '有降噪耳机推荐吗\n帮我把订单 20260701001 改价到 900 元\n确认\n退出\n' \
  | python -m shopharness.cli --mock
```

未安装本地 bge 模型或向量依赖时，4 项向量集成测试会跳过；关键词检索和降级测试仍会运行。

## 云端 API 模式

使用支持 OpenAI Chat Completions 和 function calling 的云端模型。无需下载模型或启动 vLLM 服务。

首次配置时复制模板；如果已经有 `.env`，直接编辑现有文件，不要覆盖：

```bash
cp -n .env.example .env
python -m shopharness.cli
```

启动前填写 `.env`：

```dotenv
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_API_KEY=your-api-key
MODEL=your-model-or-endpoint-id
```

`MODEL` 填写服务支持的模型 ID 或推理接入点 ID；`LLM_BASE_URL` 填写基础地址，不包含 `/chat/completions`。
火山方舟调用可参考[官方 SDK 示例](https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkexamples/volcenginesdkarkruntime/completions.py)。

| 启动方式 | 模型来源 |
| --- | --- |
| `python -m shopharness.cli` | 工作目录下的 `.env` 或进程环境变量；没有云端配置时使用 Mock |
| `python -m shopharness.cli --mock` | 强制 Mock，忽略云端配置 |
| `python -m shopharness.cli --endpoint http://localhost:8000/v1` | 显式连接 vLLM，忽略云端地址、密钥和模型配置 |

进程环境变量优先于 `.env`；`--model` 覆盖当前模式的模型。云端配置部分缺失会报错。
`--mock` 与 `--endpoint` 互斥。`--thinking` 仅控制 vLLM，云端使用服务端默认思考设置。
客户端错误不回显响应体或云端密钥；`.env` 不纳入版本控制。

主代理和子代理共享所选客户端。训练、轨迹采集及自进化脚本保持原有 Mock/vLLM 行为，不自动读取云端配置。

## 云端向量检索(百炼 OpenAI 兼容接口)

在 `.env` 中添加独立向量配置，使用基础安装即可，无需安装 vLLM、PyTorch、Transformers 或下载 bge：

```dotenv
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_MODEL=text-embedding-v4
```

`EMBEDDING_BASE_URL` 使用与密钥地域匹配的 OpenAI 兼容基础地址，不附加 `/embeddings`。
调用方式参考[百炼官方 OpenAI Embedding 文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)：
使用 `client.embeddings.create`、`encoding_format="float"`，每批最多 10 条文本，使用模型默认维度。
客户端按响应 `index` 恢复顺序，校验维度和数值并进行 L2 归一化。

- 向量配置与对话模型的地址、密钥和模型名分别读取，不共用凭据。进程环境变量优先于 `.env`。
- 兼容现有拼写 `EMBEDDDING_MODEL`；同一配置来源中，标准名称 `EMBEDDING_MODEL` 优先。
- 普通启动与 `--endpoint` 模式均可使用云端向量；`--endpoint` 仍忽略云端**对话**配置。`--mock` 跳过云端对话和云端向量配置，保持离线演示。
- 未配置云端向量时尝试可选本地 bge，缺少模型时使用关键词检索；部分云端向量配置缺失则在启动时报错。
- 首次启动会为商品介绍和 FAQ 建立 SQLite 缓存；后续启动只更新新增、修改的文档，并清除已删除文档的向量。
- 缓存记录后端、地址、模型的身份及维度。切换模型或遇到旧的无元数据缓存时重新构建，全部成功后替换，不混合不同模型的向量。
- 初始化或查询失败会输出不含密钥的提示并回退关键词检索；原有缓存不会因一次 API 构建失败被清空。

```bash
python -m shopharness.cli
```

## 可选本地服务(vLLM + Qwen3-8B-FP8)

连接已经运行的 vLLM 服务只需基础包；只有运行服务端才需要安装 `.[vllm]`。
建议在独立的 GPU 服务环境安装该扩展，并按所选 vLLM 版本满足其 Python、操作系统和 CUDA 要求。
核心项目支持 Python 3.10 不代表 vLLM 服务端也支持该版本。服务脚本使用服务环境中的 `.venv` 路径。

```bash
# 1. 下载模型(ModelScope,约 9GB)
python scripts/download_model.py

# 2. 安装 vLLM 并启动服务(RTX 4060 Ti 16GB 验证通过)
pip install -e '.[vllm]'
bash scripts/serve_vllm.sh        # 监听 :8000,hermes tool parser + qwen3 reasoning parser

# 3. 对话(默认 /no_think 压低首 token 延迟;--thinking 开启思考模式)
python -m shopharness.cli --endpoint http://localhost:8000/v1
```

> RAG(语义检索)为可选增强:下载 bge 向量模型后自动启用,缺失时降级为关键词检索
> ```bash
> pip install -e '.[rag]'
> pip install torch  # 在需要本地向量检索的环境中单独安装适合平台的 PyTorch
> python -c "from modelscope import snapshot_download; \
>   snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')"
> ```

> 已在本机(RTX 4060 Ti 16GB / vLLM 0.26.0)实测:商品咨询、订单查询、
> 改价拦截→确认→成功、改价低于限价被护栏拒绝、检索子代理委托、
> 买家记忆跨会话注入,六条真实会话全部通过;
> 首 token 延迟约 1.6s,prefix cache 命中率 77%。
>
> 排障笔记(见 `scripts/serve_vllm.sh`):
> - 系统无 gcc → triton/torch.compile 需要 `CC`(脚本已指向 conda 工具链,
>   `conda install gcc_linux-64` 即可);无编译器时也可 `--enforce-eager`
> - 系统无 nvcc → 必须 `VLLM_USE_FLASHINFER_SAMPLER=0` 关闭 flashinfer JIT 采样器

## 商品数据与我的订单

内置 **50 件虚构商品**，覆盖数码影音、电脑外设、家居生活、服饰鞋包、生活电器、食品酒水、母婴玩具、运动户外和美妆个护。
启动时会把旧数据库缺少的种子商品补齐，保留已有价格、库存、订单备注与金额；自定义商品也不会被删除。
商品和 FAQ 的云端向量缓存会增量更新。

新增只读工具 `list_orders()`，无需订单号，查询当前会话买家的全部订单，返回订单号、商品、数量、金额、状态和创建时间。
内置三笔演示订单归属默认账号 `buyer-demo`；原来的演示姓名和收货信息保留。
历史数据库自动补充 `orders.buyer_id`，只关联已知演示订单，其他未归属的历史订单不会出现在任何买家的列表中。

```bash
# 使用 .env 中配置的云端服务，默认买家 buyer-demo
python -m shopharness.cli

# 完全离线查看演示订单
python -m shopharness.cli --mock --buyer buyer-demo

# 独立买家：没有关联订单时正常返回“暂无订单”
python -m shopharness.cli --mock --buyer buyer-new
```

可以直接输入“查询我的当前订单”“查询我当前的所有订单”或“我想查询我的所有订单”。
客服先调用 `list_orders` 列出订单，选定订单后再用 `get_order` 或 `get_logistics` 查看详情。
订单列表工具的买家身份由应用绑定，不接受模型传入买家 ID；CLI 的 `--buyer` 是演示身份，接入真实渠道时应由已认证会话提供。

`models/`、`traces/` 和 `evolve/out/` 使用 `.gitkeep` 保留目录；模型权重、会话记录、训练导出和 SQLite 缓存仍被忽略。

## 一次真实会话长什么样

```
买家: 有降噪耳机推荐吗
  🎯 [skill_activated] inquiry-conversion
  🔧 [tool_call] search_products({"keyword": "耳机"})
  ✅ [tool_result] search_products 成功
客服: 为您查到 YX-1001 音弦无线降噪耳机 Pro…

买家: 帮我把订单 20260701001 改价到 900 元
  🛑 [dangerous_intercepted] adjust_price      ← 危险操作确认门
客服: 确认一下:您希望将订单 20260701001 的金额改为 900 元…请回复「确认」。

买家: 确认
  👍 [confirmed] 买家确认执行 adjust_price
  🔧 [tool_call] adjust_price(...)
  ✅ [tool_result] adjust_price 成功
```

## 相对完整设计的取舍

| DESIGN.md 规划 | 本版实现 | 升级路径 |
| --- | --- | --- |
| Postgres + pgvector | SQLite + 关键词检索 | 换连接串 + pgvector 索引,接口不变 |
| 独立 MCP Server 进程 | 进程内 ToolRegistry(schema 同 MCP 风格) | FastMCP 包装 tools/servers.py,Streamable HTTP 暴露 |
| OTel + Langfuse | JSONL trace(字段已对齐 GenAI semconv) | Tracer.span 替换为 OTel SDK exporter |
| 子代理 / LangGraph 长流程 | ✅ 已实现(M3) | — |
| 自进化 / 数据飞轮 | ✅ 已实现闭环、导出与 QLoRA SFT 实测(留出集 5/6→6/6) | 数据规模化后接 DPO/GRPO(Agentic RL) |

## M3/M4 使用说明

```bash
# 子代理(主代理工具表中已注册 delegate_research / delegate_aftersale)
python -m shopharness.cli --mock
> YX-1001 和 YX-1003 对比哪个好      # 触发检索子代理,主上下文只见摘要

# 售后长流程(LangGraph,演示中断与跨进程恢复)
python -m shopharness.cli --flow aftersale

# 分层记忆(按买家 ID 沉淀,再次进入自动注入 L1)
python -m shopharness.cli --mock --buyer 张三

# 自进化闭环(默认 dry-run:只出 bad case 报告与提案)
python -m evolve.run_cycle
python -m evolve.run_cycle --apply     # 完整闭环:门禁不过自动回滚

# 数据飞轮导出(脱敏 + 校验)
python evolve/export_sft.py            # traces → evolve/out/sft.jsonl
python evolve/export_dpo.py            # traces → evolve/out/dpo.jsonl

# Post-training 闭环(已实测跑通):
python evolve/collect_sft.py           # 真实 vLLM 轨迹采集(拒绝采样)
python evolve/train_lora.py            # QLoRA SFT(4bit nf4 + LoRA r16,16GB 显存)
python evolve/merge_lora.py            # adapter 合并回 BF16 基座
bash scripts/serve_sft.sh                        # 部署微调模型(动态 FP8 量化)
python evolve/eval_lora.py --model cs-sft   # 留出集对比
# 实测:留出集 trajectory 通过率 基线 5/6 → 微调后 6/6
# (失败案例"精华到手价"微调后正确选择 calc_discount)
```

## 目录结构

```
shopharness/        # harness 包(core / llm / tools / flows / data / cli)
skills/             # SKILL.md 技能(可热加载)
evolve/             # 自进化闭环 + SFT/DPO 数据飞轮导出(M4)
eval/               # 18 条 trajectory 评测场景(含 --gate 回归门禁)
tests/              # Mock / HTTP 模拟测试,另含 4 项可选本地向量集成测试
scripts/            # 模型下载 + vLLM 启动
traces/             # 运行生成的 JSONL trace(gitignore)
models/             # Qwen3-8B-FP8 权重(gitignore)
```
