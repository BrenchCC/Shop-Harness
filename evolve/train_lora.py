"""QLoRA SFT 训练:Qwen3-8B-FP8 基座 + 4bit 量化 + LoRA,16GB 显存可跑。

用法(需先停掉 vLLM 释放显存):
    .venv/bin/python evolve/train_lora.py [--data evolve/out/sft_full.jsonl]
输出: evolve/out/lora-cs/(adapter_model.safetensors 等)

说明:
- 基座为 FP8 checkpoint,加载时 dequant 到 bf16 再由 bnb 量化为 4bit
- 优先使用 chat template 的 assistant token mask(completion-only loss),
  模板不支持时回退全序列 loss 并打印警告
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

MODEL_PATH = "models/Qwen3-8B-bf16-true"  # FP8 手动反量化得到的 BF16 基座
DEFAULT_OUT = "evolve/out/lora-cs"


def load_dataset_jsonl(path: str) -> Dataset:
    rows = [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)


def _flat(ids):
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(t) for t in ids]


def _assistant_spans(tokenizer, messages) -> list[tuple[int, int]]:
    """用增量模板渲染定位所有 assistant 段(含 tool_calls)的 token 区间。"""
    spans = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        pre = _flat(tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True))
        full_i = _flat(tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True))
        if full_i[: len(pre)] == pre and len(full_i) > len(pre):
            spans.append((len(pre), len(full_i)))
    return spans


def build_features(tokenizer, sample: dict, max_len: int) -> dict:
    messages = sample["messages"]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, truncation=True,
            max_length=max_len, return_assistant_tokens_mask=True)
        input_ids = encoded["input_ids"]
        mask = encoded["assistant_masks"]
        # BatchEncoding 可能带 batch 维度,拍平成一维
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = list(input_ids[0])
        if mask and isinstance(mask[0], (list, tuple)):
            mask = list(mask[0])
        input_ids = [int(t) for t in input_ids]
        mask = [int(m) for m in mask]
        labels = [tok if m == 1 else -100
                  for tok, m in zip(input_ids, mask)]
        if all(label == -100 for label in labels):
            raise ValueError("empty assistant mask")
    except (TypeError, KeyError, ValueError) as exc:
        if not hasattr(build_features, "_warned"):
            print(f"[warn] assistant mask 不可用({exc}),"
                  "回退: 手工 span mask(训练全部 assistant 段含 tool_calls)")
            build_features._warned = True
        input_ids = _flat(tokenizer.apply_chat_template(
            messages, tokenize=True, truncation=True, max_length=max_len))
        labels = [-100] * len(input_ids)
        for start, end in _assistant_spans(tokenizer, messages):
            end = min(end, len(input_ids))
            labels[start:end] = input_ids[start:end]
        if all(label == -100 for label in labels):
            labels = list(input_ids)  # 极端兜底:全序列 loss
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="evolve/out/sft_full.jsonl")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=4096)
    args = parser.parse_args()

    print(f"加载 tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("加载基座(4bit QLoRA,FP8 → bf16 → nf4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_dataset_jsonl(args.data)
    print(f"样本数: {len(dataset)}")
    features = dataset.map(
        lambda s: build_features(tokenizer, s, args.max_len),
        remove_columns=dataset.column_names)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=features,
                      data_collator=Collator(tokenizer.pad_token_id or 0))
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"LoRA adapter 已保存: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
