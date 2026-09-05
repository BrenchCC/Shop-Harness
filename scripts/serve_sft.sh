#!/usr/bin/env bash
# 部署 LoRA 合并后的微调模型(models/Qwen3-8B-cs-sft)
# BF16 合并权重由 vLLM 运行时动态 FP8 量化,适配 16GB 显存
set -euo pipefail
cd "$(dirname "$0")/.."

export CC="${CC:-$HOME/miniconda3/bin/x86_64-conda-linux-gnu-cc}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec .venv/bin/vllm serve models/Qwen3-8B-cs-sft \
  --served-model-name cs-sft \
  --quantization fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port 8000
