# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShopHarness is an **Agent Harness** (scaffold) that wraps a cloud API or an optionally locally-deployed Qwen3-8B model to deliver production-grade e-commerce customer service (售前/售中/售后). The thesis: models are commoditized, the harness is the moat. It layers context engineering, tool calling, permission/safety, subagents, memory, and a self-evolution loop on top of a small open model.

The repo implements DESIGN.md's M1+M2 core plus M3 (subagents + LangGraph flows) and M4 (memory + self-evolution + data flywheel). `DESIGN.md` is the full architecture; `README.md` maps it to code; `RESUME.md` is interview Q&A. All prose, docstrings, and comments in the source are written in **Chinese** — match that style in new code.

The whole system runs without a GPU in **Mock mode** (a scripted `MockLLM`), which is what all tests and eval use. Real inference uses cloud Chat Completions or an explicitly selected vLLM endpoint. Cloud mode does not require the vLLM server package.

## Commands

### Setup

```bash
pip install -e '.[dev]'  # Python >=3.10; does not install vLLM
```

(Network-constrained envs use the Aliyun mirror. `pyproject.toml` declares deps; install extras via `pip install -e '.[rag]'` / `'.[vllm]'` / `'.[dev]'` as needed.)

### Tests

```bash
python -m pytest            # full suite (run from repo root; `-q` + testpaths are in pyproject)
python -m pytest tests/test_harness.py -k <test_name>   # single test
```

Core tests use MockLLM or simulated HTTP without network access. Four optional vector integration tests skip when the local bge model or inference dependencies are absent. `tests/conftest.py` builds a `Harness` via `build_harness()` with `rag_enabled=False` and a tmp SQLite DB. It also exports `event_types()` and `tool_calls()` helpers for asserting on `TurnResult.events`.

### Eval (trajectory scenarios)

```bash
python eval/run_eval.py          # 18 scripted scenarios, pass/fail matrix
python eval/run_eval.py --gate   # exit-code variant used as CI/self-evolution gate
```

Scenarios assert tool-call subsequence, event stream (intercept/guardrail/circuit-break/compaction/handoff), DB final state, and L2 fact retention — not single outputs.

### Demo / run

```bash
printf '有降噪耳机推荐吗\n帮我把订单 20260701001 改价到 900 元\n确认\n退出\n' \
  | python -m shopharness.cli --mock      # full "改价确认" plot
python -m shopharness.cli --flow aftersale            # LangGraph flow demo (interrupt + resume)
python -m shopharness.cli --mock --buyer 张三           # memory injection demo
```

### Real model (local vLLM + Qwen3-8B-FP8)

```bash
python scripts/download_model.py      # ModelScope download (~9GB)
bash scripts/serve_vllm.sh                       # :8000, hermes tool parser + qwen3 reasoning parser
python -m shopharness.cli --endpoint http://localhost:8000/v1
```

See `scripts/serve_vllm.sh` for the two known Linux host gotchas: missing gcc (needs `CC`/`CXX`) and missing nvcc (needs `VLLM_USE_FLASHINFER_SAMPLER=0`).

### Self-evolution / data flywheel (M4)

```bash
python -m evolve.run_cycle            # dry-run: bad-case report + proposals only
python -m evolve.run_cycle --apply    # full loop (offline gate → gray release → rollback on failure)
python evolve/export_sft.py           # traces → evolve/out/sft.jsonl (PII scrubbed)
python evolve/export_dpo.py           # traces → evolve/out/dpo.jsonl
python evolve/collect_sft.py          # reject-sampling real vLLM trajectories
python evolve/train_lora.py           # QLoRA SFT (4bit nf4 + LoRA r16)
python evolve/merge_lora.py           # merge adapter back to BF16
python evolve/eval_lora.py --model cs-sft
```

## Architecture

### Layering and data flow

```
channel (cli.py) → Harness Core (loop/context/permissions/skills/memory)
                 → tool layer (ToolRegistry → SQLite)
                 → inference (LLMClient Protocol → MockLLM | OpenAIClient → cloud | VLLMClient → vLLM)
                 → data (SQLite shop.db + JSONL traces)
```

The decoupling rule: the harness only talks to a model through the `LLMClient` Protocol and to tools only through `ToolRegistry`. Either side is independently swappable (mock ↔ cloud ↔ vLLM; in-process registry ↔ MCP server later).

### The composition root

`build_harness()` in `shopharness/cli.py` is the dependency-injection entry point — the CLI, tests (`conftest.py`), and eval (`run_eval.py`) all assemble the system through it. New collaborators should be wired there, not hardcoded in `Harness`.

### The main loop (`shopharness/core/harness.py`)

`Harness.handle(user_text)` runs one turn: handoff check → dangerous-op confirmation → skill routing (which trims the tool whitelist) → a `max_tool_steps`-bounded decision↔tool loop. Every failure path (illegal tool name, repeated tool failure/circuit breaker, step overflow, LLM error) terminates in a **handoff** with a generated summary — there is no crash/dead-loop fallback. Everything the loop does is surfaced as `TurnEvent`s (and mirrored to `Tracer` spans), which is what CLI icons, tests, and eval all consume.

### Core modules (`shopharness/core/`)

- **`context.py`** — `ContextManager` implements L0–L4 layered context and the three-level compaction (slim tool results → sliding window + fact extraction into L2 `SessionState` → full summary). Compaction is triggered by the harness against a token budget, transparent to the model. Token counts use the heuristic `estimate_tokens()` in `config.py`.
- **`harness.py`** — the loop above.
- **`permissions.py` + `hooks.py`** — READ/WRITE/DANGEROUS levels; dangerous ops (改价/退款) require buyer confirmation (the `_pending_dangerous` gate), a pre-tool guardrail (min-price rule), and audit logging. Hooks are registered in `build_harness` via `make_price_guardrail` / `make_audit_hook`.
- **`subagent.py`** — `SubagentRunner` runs a context-isolated harness loop and returns only a conclusion summary; registered into the main tool table as `delegate_*` tools.
- **`memory.py`** — three-layer memory (episodic summary / semantic buyer profile / procedural skill versions), injected as L1 at session start.
- **`rag.py`** — independent cloud embeddings or optional local bge-small-zh, with product keyword/vector RRF. SQLite caches track model identity and dimensions; changed documents refresh at startup. API failures fall back to keywords. CLI loads `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, and `EMBEDDING_MODEL` (also accepts `EMBEDDDING_MODEL`); `--mock` skips cloud configuration. Library `Settings()` never reads `.env` implicitly.
- **`trace.py`** — JSONL tracer whose span fields align with OTel GenAI semantic conventions.

### Tools (`shopharness/tools/`)

`registry.py` defines `Tool`/`ToolRegistry` (OpenAI function-calling schemas; a whitelist trims what reaches the 8B model). `servers.py` builds the 10 business tools over SQLite and the two guardrail/audit hooks. Key design point: **robustness lives in the tools, not the model** — tools self-correct bad model args (e.g. an invalid category falls back to full search with a note) rather than assuming correct parameters.

### Long-running flows (`shopharness/flows/aftersale.py`)

The aftersale ticket flow is the one place using **LangGraph** (not the main loop) — it needs `interrupt` for buyer confirmation and `SqliteSaver` checkpointing for cross-process resume. The main dialogue deliberately uses the hand-rolled loop for full control over compaction/permission interception.

### Self-evolution (`evolve/`)

`run_cycle.py` orchestrates: mine bad cases from traces → `propose.py` LLM proposals → `gate.py` runs the eval scenarios as a regression gate → `release.py` gray-releases skill versions (rollback on gate failure). `export_sft.py` / `export_dpo.py` turn traces into training JSONL with forced PII scrubbing and schema validation.

### Data

Single SQLite DB (`shopharness/data/shop.db`, schema + seed in `data/seed.py`, created lazily via `ensure_db()`): 50 fictional seed products, buyer-scoped order lists, `tickets`, `audit`, memory, and the vector store. Existing databases add missing products without overwriting edits; legacy demo orders gain `buyer_id=buyer-demo`. `list_orders()` is bound to the application-supplied buyer ID, not model arguments. `traces/` and `models/` are gitignored runtime artifacts.

## Conventions

- Comments, docstrings, and multi-line strings in code are written in **Chinese**, mirroring the existing files.
- Pydantic v2 models are used for config (`Settings`), message schemas (`Message`/`ToolCall`), and structured session state (`SessionState`).
- Unit tests never touch a real model or network and disable RAG; trigger compaction by instantiating `Settings` with a small `context_budget`.
- There are no linters or type-checkers configured; keep changes consistent with the surrounding file's style.