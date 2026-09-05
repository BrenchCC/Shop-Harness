# ShopHarness

面向电商客服场景的 **Agent Harness(脚手架)** —— 让本地部署的 Qwen3-8B 打出大模型级客服效果。
设计文档见 [DESIGN.md](DESIGN.md)(本仓库实现其 M1+M2 核心版)。

## 核心能力

| 模块 | 实现 | 位置 |
| --- | --- | --- |
| Harness 主 loop | turn 管理、最大步数、非法工具自我纠正、连续失败熔断 | `shopharness/core/harness.py` |
| 上下文工程 | 分层上下文(L0 技能指令 / L1 记忆 / L2 结构化状态 / L3+L4 历史)+ 三级 compaction | `shopharness/core/context.py` |
| 工具系统 | 9 个业务工具(SQLite),OpenAI function calling schema,按技能白名单动态裁剪 | `shopharness/tools/` |
| **RAG 检索增强** | bge-small-zh 向量语义检索 + 关键词检索,RRF 混合排序;商品库 + FAQ 知识库;模型缺失自动降级 | `shopharness/core/rag.py` |
| 权限模型 | READ / WRITE / DANGEROUS 三级;改价须经买家复述确认 + 最低限价护栏 + 审计落库 | `shopharness/core/permissions.py`、`hooks.py` |
| Skills | 目录式 SKILL.md(询单转化 / 催付 / 退换 SOP),意图路由激活,热加载 | `skills/`、`core/skills.py` |
| 转人工 | 关键词/熔断/步数超限触发,自动生成交接摘要并建工单 | `shopharness/core/handoff.py` |
| **子代理(M3)** | 上下文隔离的检索/售后子代理,仅回传结论摘要;注册为 `delegate_*` 工具 | `shopharness/core/subagent.py` |
| **长程流程(M3)** | LangGraph 售后工单流程,interrupt 等买家确认,SqliteSaver checkpoint 跨进程恢复 | `shopharness/flows/aftersale.py` |
| **分层记忆(M4)** | 情景(会话摘要)/ 语义(买家画像)/ 程序性(技能版本)三层,L1 注入 | `shopharness/core/memory.py` |
| **自进化(M4)** | bad case 挖掘 → LLM 提案 → 离线门禁 → 灰度/回滚,dry-run 默认 | `evolve/` |
| **数据飞轮(M4)** | traces → SFT/DPO JSONL 导出,PII 脱敏 + schema 校验 | `evolve/export_*.py` |
| 可观测性 | JSONL trace,字段对齐 OTel GenAI 语义约定 | `shopharness/core/trace.py` |
| 评测 | 15 条脚本化场景,trajectory 断言 + `--gate` 回归门禁 | `eval/` |

## 快速开始(Mock 模式,零 GPU 依赖)

```bash
# 环境(网络受限时走阿里云镜像)
python3 -m pip install --user -i https://mirrors.aliyun.com/pypi/simple/ uv
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -i https://mirrors.aliyun.com/pypi/simple/ \
    openai pydantic rank_bm25 pytest httpx

# 跑测试与评测
.venv/bin/python -m pytest tests/ -q      # 71 项
.venv/bin/python eval/run_eval.py         # 15 场景

# 演示对话(含完整"改价确认"剧情)
printf '有降噪耳机推荐吗\n帮我把订单 20260701001 改价到 900 元\n确认\n退出\n' \
  | .venv/bin/python -m shopharness.cli --mock
```

## 真实模式(本地 vLLM + Qwen3-8B-FP8)

```bash
# 1. 下载模型(ModelScope,约 9GB)
.venv/bin/python scripts/download_model.py

# 2. 安装 vLLM 并启动服务(RTX 4060 Ti 16GB 验证通过)
uv pip install --python .venv/bin/python -i https://mirrors.aliyun.com/pypi/simple/ vllm
bash scripts/serve_vllm.sh        # 监听 :8000,hermes tool parser + qwen3 reasoning parser

# 3. 对话(默认 /no_think 压低首 token 延迟;--thinking 开启思考模式)
.venv/bin/python -m shopharness.cli --endpoint http://localhost:8000/v1
```

> RAG(语义检索)为可选增强:下载 bge 向量模型后自动启用,缺失时降级为关键词检索
> ```bash
> uv pip install --python .venv/bin/python -i https://mirrors.aliyun.com/pypi/simple/ transformers
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
.venv/bin/python -m shopharness.cli --mock
> YX-1001 和 YX-1003 对比哪个好      # 触发检索子代理,主上下文只见摘要

# 售后长流程(LangGraph,演示中断与跨进程恢复)
.venv/bin/python -m shopharness.cli --flow aftersale

# 分层记忆(按买家 ID 沉淀,再次进入自动注入 L1)
.venv/bin/python -m shopharness.cli --mock --buyer 张三

# 自进化闭环(默认 dry-run:只出 bad case 报告与提案)
.venv/bin/python -m evolve.run_cycle
.venv/bin/python -m evolve.run_cycle --apply     # 完整闭环:门禁不过自动回滚

# 数据飞轮导出(脱敏 + 校验)
.venv/bin/python evolve/export_sft.py            # traces → evolve/out/sft.jsonl
.venv/bin/python evolve/export_dpo.py            # traces → evolve/out/dpo.jsonl

# Post-training 闭环(已实测跑通):
.venv/bin/python evolve/collect_sft.py           # 真实 vLLM 轨迹采集(拒绝采样)
.venv/bin/python evolve/train_lora.py            # QLoRA SFT(4bit nf4 + LoRA r16,16GB 显存)
.venv/bin/python evolve/merge_lora.py            # adapter 合并回 BF16 基座
bash scripts/serve_sft.sh                        # 部署微调模型(动态 FP8 量化)
.venv/bin/python evolve/eval_lora.py --model cs-sft   # 留出集对比
# 实测:留出集 trajectory 通过率 基线 5/6 → 微调后 6/6
# (失败案例"精华到手价"微调后正确选择 calc_discount)
```

## 目录结构

```
shopharness/        # harness 包(core / llm / tools / flows / data / cli)
skills/             # SKILL.md 技能(可热加载)
evolve/             # 自进化闭环 + SFT/DPO 数据飞轮导出(M4)
eval/               # 15 条 trajectory 评测场景(含 --gate 回归门禁)
tests/              # 64 项 pytest(全部 Mock,零外部依赖)
scripts/            # 模型下载 + vLLM 启动
traces/             # 运行生成的 JSONL trace(gitignore)
models/             # Qwen3-8B-FP8 权重(gitignore)
```
