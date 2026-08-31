#!/usr/bin/env python3
"""Reproduce the reported full-parameter Qwen3-8B judge SFT run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


class SFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.rows[index]
        messages = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        user_text = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)[:self.max_length]
        user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
        labels = full_ids.copy()
        labels[:min(len(user_ids), len(labels))] = [-100] * min(len(user_ids), len(labels))
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones_like(input_ids),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-steps", type=int, default=200)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = SFTDataset(args.train, tokenizer, args.max_length)
    dev_data = SFTDataset(args.dev, tokenizer, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={"fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer"},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    final_dir = args.output / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
