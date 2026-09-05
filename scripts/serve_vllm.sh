#!/usr/bin/env bash
# 启动本地 vLLM 服务(Qwen3-8B-FP8,RTX 4060 Ti 16GB 验证通过配置)
set -euo pipefail
cd "$(dirname "$0")/.."

# triton/torch.compile 需要宿主 C 编译器(系统无 gcc,用 conda 装的工具链)
export CC="${CC:-$HOME/miniconda3/bin/x86_64-conda-linux-gnu-cc}"
export CXX="${CXX:-$HOME/miniconda3/bin/x86_64-conda-linux-gnu-c++}"
# 系统无 nvcc,关闭 flashinfer JIT 采样器(退回 PyTorch 原生采样)
export VLLM_USE_FLASHINFER_SAMPLER=0

exec .venv/bin/vllm serve models/Qwen3-8B-FP8 \
  --served-model-name Qwen/Qwen3-8B-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port 8000
