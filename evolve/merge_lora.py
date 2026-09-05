"""把 LoRA adapter 合并回 BF16 基座,产出可独立部署的微调模型。

用法:
    .venv/bin/python evolve/merge_lora.py [--adapter evolve/out/lora-cs]
输出: models/Qwen3-8B-cs-sft/(完整模型,safetensors)
合并后可由 vLLM 以 --quantization fp8 动态量化部署(16GB 显存)。
"""

from __future__ import annotations

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "models/Qwen3-8B-bf16-true"
MERGED = "models/Qwen3-8B-cs-sft"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="evolve/out/lora-cs")
    parser.add_argument("--out", default=MERGED)
    args = parser.parse_args()

    print("CPU 加载 BF16 基座(约 16GB 内存)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"挂载 adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("合并中...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.save_pretrained(args.out)
    print(f"微调模型已保存: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
